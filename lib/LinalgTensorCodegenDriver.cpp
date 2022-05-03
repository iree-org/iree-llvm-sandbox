//===- LinalgTensorCodegenDriver.cpp - Linalg transformation driver--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Passes/PassDetail.h"
#include "Passes/Passes.h"

#include "Passes/Transforms.h"

#include "Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "Dialect/LinalgExt/LinalgExtBufferization.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

static void
getAtMostNEnclosingLoops(Operation *op, int64_t nLoops,
                         SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = op->getParentOp();
  while (nLoops-- > 0 &&
         (outermostEnclosingForOp = dyn_cast<scf::ForOp>(nextEnclosingOp))) {
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingOp = outermostEnclosingForOp->getParentOp();
  }
}

struct LinalgFusePass : public LinalgFuseBase<LinalgFusePass> {
  LinalgFusePass() = default;
  LinalgFusePass(const LinalgFusePass &pass) {}
  void runOnOperation() override;
};

struct LinalgFuseOutputIntoReductionPass
    : public LinalgFuseOutputIntoReductionBase<
          LinalgFuseOutputIntoReductionPass> {
  LinalgFuseOutputIntoReductionPass() = default;
  LinalgFuseOutputIntoReductionPass(
      const LinalgFuseOutputIntoReductionPass &pass) {}
  void runOnOperation() override;
};

struct UnrollOneVectorOpPass
    : public UnrollOneVectorOpBase<UnrollOneVectorOpPass> {
  UnrollOneVectorOpPass() = default;
  UnrollOneVectorOpPass(const UnrollOneVectorOpPass &pass) {}
  void runOnOperation() override;
};

} // namespace

/// Collect all Linalg ops, they must all have tensor semantics.
/// For now this just fuses everything.
// TODO: finer control.
void LinalgFusePass::runOnOperation() {
  FuncOp funcOp = getOperation();
  if (anchorOpName.empty())
    return;

  // Set up tiling and vectorization options.
  LinalgTilingAndFusionOptions tilingOptions;
  tilingOptions.tileSizes = {tileSizes.begin(), tileSizes.end()};
  tilingOptions.tileInterchange = {tileInterchange.begin(),
                                   tileInterchange.end()};

  // Parse the padding values.
  SmallVector<Attribute> paddingValueAttributes;
  for (const std::string &paddingValue : paddingValues) {
    paddingValueAttributes.push_back(
        parseAttribute(paddingValue, &getContext()));
  }

  // Set up padding options.
  SmallVector<SmallVector<int64_t>> transposePaddingVectors;
  for (const std::string &transposePadding : transposePaddings) {
    SmallVector<int64_t> transposeVector = {};
    SmallVector<StringRef> tokens;
    StringRef(transposePadding).split(tokens, ':');
    for (StringRef token : tokens)
      transposeVector.push_back(std::stoi(token.str()));
    transposePaddingVectors.push_back(transposeVector);
  }

  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValues(paddingValueAttributes);
  paddingOptions.setPaddingDimensions(
      SmallVector<int64_t>{paddingDimensions.begin(), paddingDimensions.end()});
  paddingOptions.setPackPaddings(
      SmallVector<bool>{packPaddings.begin(), packPaddings.end()});
  paddingOptions.setHoistPaddings(
      SmallVector<int64_t>{hoistPaddings.begin(), hoistPaddings.end()});
  paddingOptions.setTransposePaddings(transposePaddingVectors);

  CodegenStrategy strategy;
  strategy.tileAndFuseIf(!tileSizes.empty(), anchorOpName, tilingOptions)
      .padIf(pad, "", paddingOptions)
      .vectorizeIf(vectorize, "", nullptr, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());

  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}

void LinalgFuseOutputIntoReductionPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  if (funcOp.getName() != anchorFuncOpName)
    return;

  mlir::RewritePatternSet patterns(funcOp.getContext());
  populateFuseFillIntoReductionPatterns(patterns);
  (void)mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

void UnrollOneVectorOpPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName)
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShape(targetShape)
                    .setFilterConstraint([&](Operation *op) {
                      auto unrollInterface =
                          dyn_cast<VectorUnrollOpInterface>(op);
                      if (!unrollInterface ||
                          op->getName().getStringRef() != anchorOpName ||
                          !sourceShape.hasValue() ||
                          !unrollInterface.getShapeForUnroll().hasValue())
                        return failure();

                      ArrayRef<int64_t> sourceShapeToMatch{sourceShape};
                      auto shapeForUnroll =
                          unrollInterface.getShapeForUnroll().getValue();
                      ArrayRef<int64_t> actualSourceShape{
                          shapeForUnroll.begin(), shapeForUnroll.end()};
                      return success(sourceShapeToMatch == actualSourceShape);
                    }));
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

static scf::ExecuteRegionOp outlineInExecuteRegion(RewriterBase &b,
                                                   Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

// Naive schedule: Schedule ops as early as possible.
static void
loopScheduling(scf::ForOp forOp,
               std::vector<std::pair<Operation *, unsigned>> &schedule,
               unsigned II, unsigned readLatency) {
  auto getLatency = [&](Operation *op) {
    if (isa<vector::TransferReadOp>(op))
      return readLatency;
    return unsigned(1);
  };

  DenseMap<Operation *, unsigned> opCycles;
  std::map<unsigned, std::vector<Operation *>> wrappedSchedule;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    unsigned earlyCycle = 0;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      earlyCycle = std::max(earlyCycle, opCycles[def] + getLatency(def));
    }
    opCycles[&op] = earlyCycle;
    wrappedSchedule[earlyCycle % II].push_back(&op);
  }
  for (auto it : wrappedSchedule) {
    for (Operation *op : it.second) {
      unsigned cycle = opCycles[op];
      schedule.push_back(std::make_pair(op, cycle / II));
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass creation entry points.
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgFusePass() {
  return std::make_unique<LinalgFusePass>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgFuseOutputIntoReductionPass() {
  return std::make_unique<LinalgFuseOutputIntoReductionPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createUnrollOneVectorOpPass() {
  return std::make_unique<UnrollOneVectorOpPass>();
}
