//===- VectorWarpUtils.cpp - Utilities vector warp ops --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorExtOps.h"
#include "Dialects/VectorExt/VectorExtWarpUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::vector_ext;

static const int kWarpSize = 32;

// Clones `op` into a new operations that takes `operands` and returns
// `resultTypes`.
static Operation *cloneOpWithOperandsAndTypes(OpBuilder &builder, Location loc,
                                              Operation *op,
                                              ArrayRef<Value> operands,
                                              ArrayRef<Type> resultTypes) {
  OperationState res(loc, op->getName().getStringRef(), operands, resultTypes,
                     op->getAttrs());
  return builder.createOperation(res);
}

/// Helper to create a new WarpSingleLaneOp regions with different signature.
static WarpSingleLaneOp
moveRegionToNewWarpOpAndReplaceReturns(OpBuilder &b, WarpSingleLaneOp warpOp,
                                       ValueRange newYieldedValues,
                                       TypeRange newReturnTypes) {
  // Create a new op before the existing one, with the extra operands.
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(warpOp);
  auto newWarpOp = b.create<WarpSingleLaneOp>(warpOp.getLoc(), newReturnTypes,
                                              warpOp.laneid());

  Region &opBody = warpOp.getBodyRegion();
  Region &newOpBody = newWarpOp.getBodyRegion();
  newOpBody.takeBody(opBody);
  auto yield =
      cast<vector_ext::YieldOp>(newOpBody.getBlocks().begin()->getTerminator());
  yield.operandsMutable().assign(newYieldedValues);
  return newWarpOp;
}

/// Helper to create a new WarpSingleLaneOp region with extra outputs.
static WarpSingleLaneOp
moveRegionToNewWarpOpAndAppendReturns(OpBuilder &b, WarpSingleLaneOp warpOp,
                                      ValueRange newYieldedValues,
                                      TypeRange newReturnTypes) {

  SmallVector<Type> types(warpOp.getResultTypes().begin(),
                          warpOp.getResultTypes().end());
  types.append(newReturnTypes.begin(), newReturnTypes.end());
  auto yield = cast<vector_ext::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  SmallVector<Value> yieldValues(yield.getOperands().begin(),
                                 yield.getOperands().end());
  yieldValues.append(newYieldedValues.begin(), newYieldedValues.end());
  WarpSingleLaneOp newWarpOp =
      moveRegionToNewWarpOpAndReplaceReturns(b, warpOp, yieldValues, types);
  for (auto it :
       llvm::zip(warpOp.getResults(),
                 newWarpOp.getResults().take_front(warpOp.getNumResults())))
    std::get<0>(it).replaceAllUsesWith(std::get<1>(it));
  return newWarpOp;
}

OpOperand *getWarpResult(WarpSingleLaneOp warpOp,
                         std::function<bool(Operation *)> fn) {
  auto yield = cast<vector_ext::YieldOp>(
      warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
  for (OpOperand &yieldOperand : yield->getOpOperands()) {
    Value yieldValues = yieldOperand.get();
    Operation *definedOp = yieldValues.getDefiningOp();
    if (definedOp && fn(definedOp)) {
      if (!warpOp.getResult(yieldOperand.getOperandNumber()).use_empty())
        return &yieldOperand;
    }
  }
  return {};
}

/// Currently the distribution map is implicit based on the vector shape. In the
/// future it will be part of the op.
/// Example:
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1x16x2xf32>) {
///   ...
///   vector_ext.yield %3 : vector<32x16x64xf32>
/// }
/// ```
/// Would have an implicit map of:
/// `(d0, d1, d2) -> (d0, d2)`
static AffineMap calculateImplicitMap(Value yield, Value ret) {
  auto srcType = yield.getType().cast<VectorType>();
  auto dstType = ret.getType().cast<VectorType>();
  SmallVector<AffineExpr, 4> perm;
  // Check which dimension have a multiplicity greater than 1 and associated
  // them to the IDs in order.
  for (unsigned i = 0, e = srcType.getRank(); i < e; i++) {
    if (srcType.getDimSize(i) != dstType.getDimSize(i))
      perm.push_back(getAffineDimExpr(i, yield.getContext()));
  }
  auto map = AffineMap::get(srcType.getRank(), 0, perm, yield.getContext());
  return map;
}

/// Sink out elementwise op feeding into a warp op yield.
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
///   %3 = arith.addf %1, %2 : vector<32xf32>
///   vector_ext.yield %3 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %r:3 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %4 = arith.addf %2, %3 : vector<32xf32>
///   vector_ext.yield %4, %2, %3 : vector<32xf32>, vector<32xf32>,
///   vector<32xf32>
/// }
/// %0 = arith.addf %r#1, %r#2 : vector<1xf32>
struct WarpOpElementwise : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand = getWarpResult(warpOp, [](Operation *op) {
      return OpTrait::hasElementwiseMappableTraits(op);
    });
    if (!yieldOperand)
      return failure();
    Operation *elementWise = yieldOperand->get().getDefiningOp();
    unsigned operandIndex = yieldOperand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);
    SmallVector<Value> yieldValues;
    SmallVector<Type> retTypes;
    for (OpOperand &operand : elementWise->getOpOperands()) {
      auto targetType = VectorType::get(
          distributedVal.getType().cast<VectorType>().getShape(),
          operand.get().getType().cast<VectorType>().getElementType());
      retTypes.push_back(targetType);
      yieldValues.push_back(operand.get());
    }
    WarpSingleLaneOp newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    SmallVector<Value> newOperands(elementWise->getOperands().begin(),
                                   elementWise->getOperands().end());
    for (unsigned i : llvm::seq(unsigned(0), elementWise->getNumOperands())) {
      newOperands[i] = newWarpOp.getResult(i + warpOp.getNumResults());
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(newWarpOp);
    Operation *newOp = cloneOpWithOperandsAndTypes(
        rewriter, warpOp.getLoc(), elementWise, newOperands,
        {warpOp.getResult(operandIndex).getType()});
    newWarpOp.getResult(operandIndex).replaceAllUsesWith(newOp->getResult(0));
    rewriter.eraseOp(warpOp);
    return success();
  }
};

/// Sink out transfer_read op feeding into a warp op yield.
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>) {
///   ...
//    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
//    vector<32xf32>
///   vector_ext.yield %2 : vector<32xf32>
/// }
/// ```
/// To
/// ```
/// %dead = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<1xf32>,
/// vector<1xf32>, vector<1xf32>) {
///   ...
///   %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>,
///   vector<32xf32> vector_ext.yield %2 : vector<32xf32>
/// }
/// %0 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<1xf32>
struct WarpOpTransferRead : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::TransferReadOp>(op); });
    if (!operand)
      return failure();
    auto read = operand->get().getDefiningOp<vector::TransferReadOp>();
    unsigned operandIndex = operand->getOperandNumber();
    Value distributedVal = warpOp.getResult(operandIndex);

    SmallVector<Value, 4> indices(read.indices().begin(), read.indices().end());
    AffineMap map = calculateImplicitMap(read.getResult(), distributedVal);
    AffineMap indexMap = map.compose(read.permutation_map());
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(warpOp);
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(read.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      int64_t scale =
          distributedVal.getType().cast<VectorType>().getDimSize(vectorPos);
      indices[indexPos] =
          makeComposedAffineApply(rewriter, read.getLoc(), d0 + scale * d1,
                                  {indices[indexPos], warpOp.laneid()});
    }
    Value newRead = rewriter.create<vector::TransferReadOp>(
        read.getLoc(), distributedVal.getType(), read.source(), indices,
        read.permutation_mapAttr(), read.padding(), read.mask(),
        read.in_boundsAttr());
    distributedVal.replaceAllUsesWith(newRead);
    return success();
  }
};

/// Remove any result that has no use along with the matching yieldOp operand.
// TODO: Move this in WarpSingleLaneOp canonicalization.
struct WarpOpDeadResult : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldValues;
    auto yield = cast<vector_ext::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        continue;
      resultTypes.push_back(result.getType());
      yieldValues.push_back(yield.getOperand(result.getResultNumber()));
    }
    if (yield.getNumOperands() == yieldValues.size())
      return failure();
    WarpSingleLaneOp newWarpOp = moveRegionToNewWarpOpAndReplaceReturns(
        rewriter, warpOp, yieldValues, resultTypes);
    unsigned resultIndex = 0;
    for (OpResult result : warpOp.getResults()) {
      if (result.use_empty())
        continue;
      result.replaceAllUsesWith(newWarpOp.getResult(resultIndex++));
    }
    rewriter.eraseOp(warpOp);
    return success();
  }
};

void mlir::vector_ext::populatePropagateVectorDistributionPatterns(
    RewritePatternSet &pattern) {
  pattern.add<WarpOpElementwise, WarpOpTransferRead, WarpOpDeadResult>(
      pattern.getContext());
}

static LogicalResult rewriteWarpOpToScfFor(
    RewriterBase &rewriter, WarpSingleLaneOp warpOp, Value buffer) {
  assert(warpOp.getBodyRegion().hasOneBlock() &&
         "expected WarpOp with single block");
  Block *warpOpBody = &warpOp.getBodyRegion().front();

  // Passed all checks. Start rewriting.
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = warpOp.getLoc();
  rewriter.setInsertionPoint(warpOp);

  // Create scf.if op.
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value isLane0 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 warpOp.laneid(), c0);
  auto ifOp = rewriter.create<scf::IfOp>(loc, isLane0,
                                         /*withElseRegion=*/false);
  rewriter.eraseOp(ifOp.thenBlock()->getTerminator());

  // Store vectors that are defined outside of warpOp into the scratch pad
  // buffer.
  Value loadOffset = c0;
  SmallVector<Value> bbArgReplacements;
  for (const auto &it : llvm::enumerate(warpOp.args())) {
    Value val = it.value();
    Value bbArg = warpOpBody->getArgument(it.index());

    // Store arg vector into buffer.
    rewriter.setInsertionPoint(ifOp);
    auto vectorType = val.getType().cast<VectorType>();
    int64_t storeSize = vectorType.getShape()[0];
    Value storeOffset = rewriter.create<arith::AddIOp>(loc, loadOffset,
        rewriter.create<arith::MulIOp>(loc, warpOp.laneid(),
            rewriter.create<arith::ConstantIndexOp>(loc, storeSize)));
    rewriter.create<vector::StoreOp>(loc, val, buffer, storeOffset);

    // Load bbArg vector from buffer.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    auto bbArgType = bbArg.getType().cast<VectorType>();
    Value loadOp = rewriter.create<vector::LoadOp>(loc, bbArgType, buffer,
                                                   loadOffset);
    bbArgReplacements.push_back(loadOp);

    // Compute next offset.
    rewriter.setInsertionPoint(ifOp);
    loadOffset = rewriter.create<arith::AddIOp>(loc, loadOffset,
        rewriter.create<arith::ConstantIndexOp>(loc, bbArgType.getShape()[0]));
  }

  // Move body of warpOp to ifOp.
  rewriter.mergeBlocks(warpOpBody, ifOp.thenBlock(), bbArgReplacements);

  // Rewrite terminator and compute replacements of WarpOp results.
  SmallVector<Value> replacements;
  auto yieldOp = cast<vector_ext::YieldOp>(ifOp.thenBlock()->getTerminator());
  Location yieldLoc = yieldOp.getLoc();
  Value storeOffset = c0;

  for (const auto &it : llvm::enumerate(yieldOp.operands())) {
    Value val = it.value();

    // Store yielded vector into buffer.
    rewriter.setInsertionPoint(yieldOp);
    auto vectorType = val.getType().cast<VectorType>();
    rewriter.create<vector::StoreOp>(yieldLoc, val, buffer, storeOffset);

    // Load vector from buffer (after warpOp).
    rewriter.setInsertionPointAfter(ifOp);
    auto loadedVectorType =
        warpOp->getResultTypes()[it.index()].cast<VectorType>();
    int64_t loadSize = loadedVectorType.getShape()[0];

    // loadOffset = storeOffset + laneid * loadSize
    Value loadOffset = rewriter.create<arith::AddIOp>(loc, storeOffset,
        rewriter.create<arith::MulIOp>(loc, warpOp.laneid(),
            rewriter.create<arith::ConstantIndexOp>(loc, loadSize)));
    Value loadOp = rewriter.create<vector::LoadOp>(loc, loadedVectorType,
                                                   buffer, loadOffset);
    replacements.push_back(loadOp);

    // Compute next offset.
    rewriter.setInsertionPoint(ifOp);
    Value vectorSize = rewriter.create<arith::ConstantIndexOp>(
        yieldLoc, vectorType.getShape()[0]);
    storeOffset = rewriter.create<arith::AddIOp>(yieldLoc, storeOffset,
                                                 vectorSize);
  }

  // Delete terminator and add empty scf.yield.
  rewriter.eraseOp(yieldOp);
  rewriter.setInsertionPointToEnd(ifOp.thenBlock());
  rewriter.create<scf::YieldOp>(yieldLoc);

  // Compute replacements for WarpOp results.
  rewriter.replaceOp(warpOp, replacements);

  return success();
}

struct WarpOpToScfForPattern : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    assert(warpOp.getBodyRegion().hasOneBlock() &&
           "expected WarpOp with single block");
    Block *warpOpBody = &warpOp.getBodyRegion().front();

    // Only rank 1 vectors supported at the moment. Furthermore, the dimension
    // must be divisible by the warp size. All element types must be the same.
    Type elementType;
    auto computeAllocSize = [&](ValueRange values) -> FailureOr<int64_t> {
      int64_t allocSize = 0;
      for (Value val : values) {
        auto vectorType = val.getType().cast<VectorType>();
        if (!elementType) {
          elementType = vectorType.getElementType();
        } else {
          if (elementType != vectorType.getElementType())
            return failure();
        }
        if (vectorType.getRank() != 1)
          return failure();
        int64_t dimSize = vectorType.getShape()[0];
        if (dimSize % kWarpSize != 0)
          return failure();
        allocSize += dimSize;
      }
      return allocSize;
    };

    FailureOr<int64_t> yieldAllocSize = computeAllocSize(
        cast<vector_ext::YieldOp>(warpOpBody->getTerminator()).operands());
    if (failed(yieldAllocSize))
      return failure();
    FailureOr<int64_t> argsAllocSize = computeAllocSize(
        warpOpBody->getArguments());
    if (failed(argsAllocSize))
      return failure();

    // Allocate a shared memory buffer.
    // TODO: Use the correct memory space.
    int64_t allocSize = *yieldAllocSize > *argsAllocSize
        ? *yieldAllocSize : *argsAllocSize;
    Value alloc;
    if (allocSize > 0) {
      rewriter.setInsertionPoint(warpOp);
      alloc = rewriter.create<memref::AllocOp>(
          warpOp.getLoc(), MemRefType::get({allocSize}, elementType));
    }

    // Rewrite the op.
    return rewriteWarpOpToScfFor(rewriter, warpOp, alloc);
  }
};

void mlir::vector_ext::populateWarpSingleLaneOpToScfForPattern(
    RewritePatternSet &patterns) {
  patterns.add<WarpOpToScfForPattern>(patterns.getContext());
}
