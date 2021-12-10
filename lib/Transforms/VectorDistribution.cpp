//===- VectorDistribution.cpp - Vector distribution experiements--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/PassDetail.h"
#include "Transforms/Passes.h"
#include "Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

/// Modified version of the upstream pattern handling more cases.
/// TODO: Replace upstream once it is ready.
/// Converts TransferRead op used by ExtractMap op into a smaller dimension
/// TransferRead.
/// Example:
/// ```
/// %a = vector.transfer_read %A[%c0, %c0, %c0], %cf0:
///   memref<64x64x64xf32>, vector<64x4x32xf32>
/// %e = vector.extract_map %a[%id] : vector<64x4x32xf32> to vector<2x4x1xf32>
/// ```
/// to:
/// ```
/// %id1 = affine.apply affine_map<()[s0] -> (s0 * 2)> (%id)
/// %e = vector.transfer_read %A[%id1, %c0, %id1], %cf0 :
///   memref<64x64x64xf32>, vector<2x4x1xf32>
/// ```
struct TransferReadExtractPattern
    : public OpRewritePattern<vector::ExtractMapOp> {
  TransferReadExtractPattern(MLIRContext *context)
      : OpRewritePattern<vector::ExtractMapOp>(context) {}
  LogicalResult matchAndRewrite(vector::ExtractMapOp extract,
                                PatternRewriter &rewriter) const override {
    auto read = extract.vector().getDefiningOp<vector::TransferReadOp>();
    if (!read || read.mask()) return failure();

    SmallVector<Value, 4> indices(read.indices().begin(), read.indices().end());
    AffineMap indexMap = extract.map().compose(read.permutation_map());
    unsigned idCount = 0;
    // We cannot pull down transfer_read ops as it may cross ops with
    // side-effects.
    rewriter.setInsertionPoint(read);
    // Map the vector dimension distribution to the right indices based on the
    // projection map.
    for (auto it :
         llvm::zip(indexMap.getResults(), extract.map().getResults())) {
      AffineExpr d0, d1;
      bindDims(read.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      // If this dimension is a broadcast ignore it.
      if (!indexExpr) continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale = getAffineConstantExpr(
          extract.getResultType().getDimSize(vectorPos), read.getContext());
      indices[indexPos] = makeComposedAffineApply(
          rewriter, read.getLoc(), d0 + scale * d1,
          {indices[indexPos], extract.ids()[idCount++]});
    }
    Value newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), extract.getType(), read.source(), indices,
        read.permutation_map(), read.padding(), read.in_boundsAttr());
    rewriter.replaceOp(extract, newRead);
    return success();
  }
};

/// Propagate scf.for return values in vector.extractMap into the loop region.
/// Converts:
/// ```
///  %r = scf.for %arg0 = %c0 to %c128 step %c8 iter_args(%arg1 = %init) -> (vector<128x128xf32>) {
///    ...
///    scf.yield %y : vector<128x128xf32>
//  }
///  %ext = vector.extract_map %r[%idx, %idy] : vector<128x128xf32> to vector<4x4xf32>
/// ```
/// into
/// ```
/// %e = vector.extract_map %init[%arg0, %arg1] : vector<128x128xf32> to vector<4x4xf32>
/// %2:2 = scf.for %arg5 = %c0 to %c128 step %c8 iter_args(%arg6 = %init, %arg7 = %e) -> (vector<128x128xf32>, vector<4x4xf32>) {
///   %i = vector.insert_map %arg7, %arg6[%arg0, %arg1] : vector<4x4xf32> into vector<128x128xf32>
///   ...
///   %e1 = vector.extract_map %y[%arg0, %arg1] : vector<128x128xf32> to vector<4x4xf32>
///   scf.yield %y, %e1 : vector<128x128xf32>, vector<4x4xf32>
/// }
/// %r = vector.insert_map %2#1, %2#0[%arg0, %arg1] : vector<4x4xf32> into vector<128x128xf32>
/// %ext = vector.extract_map %r[%idx, %idy] : vector<128x128xf32> to vector<4x4xf32>
/// ```
struct SCFForExtractPattern final
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter& rewriter) const override {
    SmallVector<unsigned, 8> iteratorFolded;
    SmallVector<Operation*, 8> resultOps;
    auto initArgs = llvm::to_vector<8>(forOp.getIterOperands());
    size_t initNumArgs = initArgs.size();
    // Collect all the extract.map uses and prepare adding new iter argument
    // for each of them.
    SmallVector<std::pair<vector::ExtractMapOp, int64_t>> args;
    for (const auto &it : llvm::enumerate(forOp.getResults())) {
      vector::ExtractMapOp extractMap;
      for (Operation *user : it.value().getUsers()) {
        extractMap = dyn_cast<vector::ExtractMapOp>(user);
        if (extractMap) break;
      }
      if (!extractMap) continue;
      Value operand =
          forOp->getOperand(it.index() + forOp.getNumControlOperands());
      Value newArg = rewriter.create<vector::ExtractMapOp>(
          forOp.getLoc(), extractMap.getResultType(), operand,
          extractMap.ids());
      initArgs.push_back(newArg);
      args.push_back(std::make_pair(extractMap, it.index()));
    }
    // Create a new scf.for op with extra arguments for each extract,map uses.
    if (args.empty()) return failure();
    auto newForOp =
        rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.lowerBound(),
                                    forOp.upperBound(), forOp.step(), initArgs);
    rewriter.mergeBlocks(forOp.getBody(), newForOp.getBody(),
                         newForOp.getBody()->getArguments().take_front(
                             forOp.getBody()->getNumArguments()));
    // Replace the operation by the new one.
    SmallVector<Value, 8> repResults(newForOp.getResults().begin(),
                                     newForOp.getResults().end());
    auto yieldOp = cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    auto returnValues = llvm::to_vector<8>(yieldOp.getOperands());
    // for each new argument add extract.map/insert.map ops at the region
    // boundary.
    for (std::pair<vector::ExtractMapOp, int64_t>& arg : args) {
      rewriter.setInsertionPoint(newForOp.getBody()->getTerminator());
      returnValues.push_back(rewriter.create<vector::ExtractMapOp>(
          forOp.getLoc(), arg.first.getResultType(), returnValues[arg.second],
          arg.first.ids()));
      Value newArg = newForOp.getBody()->getArgument(
          arg.second + initNumArgs + forOp.getNumInductionVars());
      Value oldArg = newForOp.getBody()->getArgument(
          arg.second + forOp.getNumInductionVars());
      rewriter.setInsertionPointToStart(newForOp.getBody());
      auto insert = rewriter.create<vector::InsertMapOp>(
          forOp.getLoc(), newArg, oldArg, arg.first.ids());
      oldArg.replaceAllUsesExcept(insert.getResult(), insert);
      rewriter.setInsertionPoint(forOp);
      repResults[arg.second] = rewriter.create<vector::InsertMapOp>(
          forOp.getLoc(), newForOp.getResult(arg.second + initNumArgs),
          newForOp.getResult(arg.second), arg.first.ids());
    }
    // Replace the yield op by one that returns only the used values.
    yieldOp.getOperation()->setOperands(returnValues);
    rewriter.replaceOp(
        forOp, ArrayRef<Value>(repResults).take_front(forOp.getNumResults()));
    return success();
  }
};

/// Pattern to remove dead code in scf.for op after incremental conversion of
/// for result values.
/// Upstreaming this pass is under progress and can be replaced once it landes:
/// https://reviews.llvm.org/D115056
struct ForOpDeadArgElimination : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {
    Block &block = forOp.region().front();
    auto yieldOp = cast<scf::YieldOp>(block.getTerminator());
    // Assume that nothing is live at the beginning and mark values as live
    // based on uses.
    DenseSet<Value> aliveValues;
    SmallVector<Value> queue;
    // Helper to mark values as live and add them to the queue of value to
    // propagate if it is the first time we detect the value as live.
    auto markLive = [&](Value val) {
      if (!forOp->isAncestor(val.getParentRegion()->getParentOp()))
        return;
      if (aliveValues.insert(val).second)
        queue.push_back(val);
    };
    // Mark all yield operands as live if the associated forOp result has any
    // use.
    for (auto &result : llvm::enumerate(forOp.getResults())) {
      if (!result.value().use_empty())
        markLive(yieldOp.getOperand(result.index()));
    }
    if (aliveValues.size() == forOp.getNumResults())
      return failure();
    // Operations with side-effects are always live. Mark all theirs operands as
    // live.
    block.walk([&](Operation *op) {
      if (!isa<scf::YieldOp, scf::ForOp>(op) && !wouldOpBeTriviallyDead(op)) {
        for (Value operand : op->getOperands())
          markLive(operand);
      }
    });
    // Propagate live property until reaching a fixed point.
    while (!queue.empty()) {
      Value value = queue.pop_back_val();
      if(auto nestedFor = value.getDefiningOp<scf::ForOp>()) {
        auto result = value.cast<OpResult>();
        OpOperand& forOperand =
            nestedFor.getOpOperandForResult(result);
        markLive(forOperand.get());
        auto nestedYieldOp =
            cast<scf::YieldOp>(nestedFor.getBody()->getTerminator());
        Value nestedYieldOperand =
            nestedYieldOp.getOperand(result.getResultNumber());
        markLive(nestedYieldOperand);
        continue;
      }
      if (Operation *def = value.getDefiningOp()) {
        for (Value operand : def->getOperands())
          markLive(operand);
        continue;
      }
      // If an argument block is live then the associated yield operand and
      // forOp operand are live.
      auto arg = value.cast<BlockArgument>();
      if(auto forOwner = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp())) {
        if (arg.getArgNumber() < forOwner.getNumInductionVars())
          continue;
        unsigned iterIdx = arg.getArgNumber() - forOwner.getNumInductionVars();
        Value yieldOperand = yieldOp.getOperand(iterIdx);
        markLive(yieldOperand);
        markLive(forOwner.getIterOperands()[iterIdx]);
      }
    }
    SmallVector<unsigned> deadArg;
    for (auto &yieldOperand : llvm::enumerate(yieldOp->getOperands())) {
      if (aliveValues.contains(yieldOperand.value()))
        continue;
      if (yieldOperand.value() == block.getArgument(yieldOperand.index() + 1))
        continue;
      deadArg.push_back(yieldOperand.index());
    }
    if (deadArg.empty())
      return failure();
    rewriter.updateRootInPlace(forOp, [&]() {
      // For simplicity we just change the dead yield operand to use the
      // associated argument and leave the operations an argument removal to
      // dead code elimination.
      for (unsigned deadArgIdx : deadArg) {
        BlockArgument arg = block.getArgument(deadArgIdx + 1);
        yieldOp.setOperand(deadArgIdx, arg);
      }
    });
    return success();
  }
};

struct PropagateVectorDistributionPass
    : public VectorPropagateDistributionBase<PropagateVectorDistributionPass> {
  PropagateVectorDistributionPass() = default;
  PropagateVectorDistributionPass(const PropagateVectorDistributionPass &pass) {
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  /// Function pass entry point.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SCFForExtractPattern, TransferReadExtractPattern>(
        &getContext());
    vector::populatePropagateVectorDistributionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    {
      RewritePatternSet cleanupPatterns(&getContext());
      cleanupPatterns.add<ForOpDeadArgElimination>(&getContext());
      (void)applyPatternsAndFoldGreedily(getOperation(),
                                         std::move(cleanupPatterns));
    }
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createPropagateVectorDistribution() {
  return std::make_unique<PropagateVectorDistributionPass>();
}
