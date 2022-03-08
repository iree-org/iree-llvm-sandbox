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
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::vector_ext;

static const unsigned kWarpSize = 32;

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
  auto newWarpOp = b.create<WarpSingleLaneOp>(
      warpOp.getLoc(), newReturnTypes, warpOp.laneid(), warpOp.args(),
      warpOp.getBody()->getArgumentTypes());

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

namespace {

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

/// A pattern that extracts vector.reduction ops from a WarpSingleLaneOp.
/// The vector is reduced in parallel. Currently limited to vector<32x...>
/// values. Every lane reduces two scalars, 5 times in a row.
/// E.g.:
/// ```
/// %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (f32) {
///   %0 = "some_def"() : () -> (vector<32xf32>)
///   %1 = vector.reduction "add", %0 : vector<32xf32> into f32
///   vector_ext.yield %1 : f32
/// }
/// ```
/// is lowered to:
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%laneid) -> (vector<1xf32>) {
///   %1 = "some_def"() : () -> (vector<32xf32>)
///   vector_ext.yield %1 : vector<32xf32>
/// }
/// %a = vector.extract %0[0] : vector<1xf32>
/// %r0, %s0 = gpu.shuffle down %e, %c16, %c32 : f32
/// %a0 = arith.addf %a, %r0 : f32
/// %r1, %s1 = gpu.shuffle down %e, %c8, %c32 : f32
/// %a1 = arith.addf %a0, %r1 : f32
/// ...
/// %r4, %s4 = gpu.shuffle down %e, %c1, %c32 : f32
/// %a4 = arith.addf %a3, %r4 : f32
/// %r = gpu.shuffle idx %a4, %c0, %c32 : f32
/// ```
struct WarpOpReduction : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *yieldOperand = getWarpResult(warpOp, [](Operation *op) {
      return isa<vector::ReductionOp>(op);
    });
    if (!yieldOperand)
      return failure();

    auto reductionOp =
        cast<vector::ReductionOp>(yieldOperand->get().getDefiningOp());
    auto vectorType = reductionOp.vector().getType().cast<VectorType>();
    // Only rank 1 vectors supported.
    if (vectorType.getRank() != 1)
      return failure();
    // Only kWarpSize vectors supported.
    if (vectorType.getShape()[0] != kWarpSize)
      return failure();
    // Only f32 and i32 element types are supported.
    if (!reductionOp.getType().isF32() &&
        !reductionOp.getType().isSignlessInteger(32))
      return failure();

    Location yieldLoc = yieldOperand->getOwner()->getLoc();

    // Return vector that will be reduced from the WarpSingleLaneOp.
    unsigned operandIndex = yieldOperand->getOperandNumber();
    SmallVector<Value> yieldValues;
    SmallVector<Type> retTypes;
    yieldValues.push_back(reductionOp.vector());
    retTypes.push_back(VectorType::get({1}, reductionOp->getResultTypes()[0]));
    WarpSingleLaneOp newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);

    // Every lane has one scalar value. These should be reduced.
    Value laneValVec = newWarpOp.getResult(warpOp.getNumResults());
    Value laneVal =
        rewriter.create<vector::ExtractOp>(yieldLoc, laneValVec, 0);

    // Parallel reduction: Every thread reduces two values. The result is
    // stored at the lower thread. Requires log_2(kWarpSize) many parallel
    // reductions.
    for (int i = kWarpSize/2; i > 0; i /= 2) {
      Value shuffled = rewriter.create<gpu::ShuffleOp>(
          reductionOp.getLoc(), laneVal, i, /*width=*/kWarpSize,
          /*mode=*/gpu::ShuffleMode::DOWN).result();
      laneVal = makeArithReduction(rewriter, reductionOp.getLoc(),
                                   reductionOp.kind(), laneVal, shuffled);
    }

    // Broadcast the result to all lanes.
    auto shuffleOp = rewriter.create<gpu::ShuffleOp>(
        yieldLoc, laneVal, /*offset=*/0, /*width=*/kWarpSize,
        /*mode=*/gpu::ShuffleMode::IDX);
    newWarpOp.getResult(operandIndex).replaceAllUsesWith(shuffleOp.result());
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
    rewriter.setInsertionPointAfter(warpOp);
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

// If an operand is directly yielded out of the region we can forward it
// directly and it doesn't need to go through the region.
struct WarpOpForwardOperand : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    SmallVector<Type> resultTypes;
    SmallVector<Value> yieldValues;
    auto yield = cast<vector_ext::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    Value valForwarded;
    unsigned resultIndex;
    for (OpOperand &operand : yield->getOpOperands()) {
      auto arg = operand.get().dyn_cast<BlockArgument>();
      if (!arg || arg.getOwner()->getParentOp() != warpOp.getOperation())
        continue;
      Value result = warpOp.getResult(operand.getOperandNumber());
      Value warpOperand = warpOp.args()[arg.getArgNumber()];
      if (result.use_empty() || result.getType() != warpOperand.getType())
        continue;
      valForwarded = warpOperand;
      resultIndex = operand.getOperandNumber();
      break;
    }
    if (!valForwarded)
      return failure();
    warpOp.getResult(resultIndex).replaceAllUsesWith(valForwarded);
    return success();
  }
};

struct WarpOpBroadcast : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    OpOperand *operand = getWarpResult(
        warpOp, [](Operation *op) { return isa<vector::BroadcastOp>(op); });
    if (!operand)
      return failure();
    unsigned int operandNumber = operand->getOperandNumber();
    auto broadcastOp = operand->get().getDefiningOp<vector::BroadcastOp>();

    auto destVecType =
        warpOp->getResultTypes()[operandNumber].cast<VectorType>();
    WarpSingleLaneOp newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, {broadcastOp.source()},
        {broadcastOp.source().getType()});
    Value broadcasted = rewriter.create<vector::BroadcastOp>(
        broadcastOp.getLoc(), destVecType,
        newWarpOp->getResult(newWarpOp->getNumResults() - 1));
    newWarpOp->getResult(operandNumber).replaceAllUsesWith(broadcasted);

    return success();
  }
};
} // namespace

/// Helper to figure out if an op has side effects or recursive side-effects.
static bool hasSideEffect(Operation &op) {
  // If we find an op with side effect before finding a transfer_write we
  // cannot hoist out the transfer write.
  if (auto memInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
    if (!memInterface.hasNoEffect())
      return true;
    if (op.hasTrait<OpTrait::HasRecursiveSideEffects>())
      return true;
  }
  if (op.hasTrait<OpTrait::HasRecursiveSideEffects>())
    return true;
  return false;
}

// TODO: Move to the op.
static unsigned distributionRatio = 32;

struct WarpOpTransferWrite : public OpRewritePattern<vector::TransferWriteOp> {
  WarpOpTransferWrite(MLIRContext *ctx, DistributionMapFn fn,
                      PatternBenefit b = 1)
      : OpRewritePattern<vector::TransferWriteOp>(ctx, b),
        distributionMapFn(fn) {}

  /// Distribute the TransferWriteOp. Only 1D distributions and vector dims that
  /// are multiples of the distribution ratio are supported at the moment.
  LogicalResult tryDistributeOp(RewriterBase &rewriter,
                                vector::TransferWriteOp writeOp,
                                WarpSingleLaneOp warpOp) const {
    AffineMap map = distributionMapFn(writeOp);
    SmallVector<int64_t> targetShape(writeOp.getVectorType().getShape().begin(),
                                     writeOp.getVectorType().getShape().end());
    assert(map.getNumResults() == 1 &&
           "multi-dim distribution not implemented yet");
    for (unsigned i = 0, e = map.getNumResults(); i < e; i++) {
      unsigned position = map.getDimPosition(i);
      if (targetShape[position] % distributionRatio != 0)
        return failure();
      targetShape[position] = targetShape[position] / distributionRatio;
    }
    VectorType targetType =
        VectorType::get(targetShape, writeOp.getVectorType().getElementType());

    SmallVector<Value> yieldValues = {writeOp.vector()};
    SmallVector<Type> retTypes = {targetType};
    WarpSingleLaneOp newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Move op outside of region: Insert clone at the insertion point and delete
    // the old op.
    auto newWriteOp = cast<vector::TransferWriteOp>(
        rewriter.clone(*writeOp.getOperation()));
    rewriter.eraseOp(writeOp);

    rewriter.setInsertionPoint(newWriteOp);
    AffineMap indexMap = map.compose(newWriteOp.permutation_map());
    Location loc = newWriteOp.getLoc();
    SmallVector<Value> indices(newWriteOp.indices().begin(),
                               newWriteOp.indices().end());
    for (auto it : llvm::zip(indexMap.getResults(), map.getResults())) {
      AffineExpr d0, d1;
      bindDims(newWarpOp.getContext(), d0, d1);
      auto indexExpr = std::get<0>(it).dyn_cast<AffineDimExpr>();
      if (!indexExpr)
        continue;
      unsigned indexPos = indexExpr.getPosition();
      unsigned vectorPos = std::get<1>(it).cast<AffineDimExpr>().getPosition();
      auto scale =
          getAffineConstantExpr(targetShape[vectorPos], newWarpOp.getContext());
      indices[indexPos] = makeComposedAffineApply(
          rewriter, loc, d0 + scale * d1, {indices[indexPos], newWarpOp.laneid()});
    }
    newWriteOp.vectorMutable().assign(newWarpOp.getResults().back());
    newWriteOp.indicesMutable().assign(indices);

    return success();
  }

  /// Extract TransferWriteOps of vector<1x> into a separate warp op.
  LogicalResult tryExtractOp(RewriterBase &rewriter,
                             vector::TransferWriteOp writeOp,
                             WarpSingleLaneOp warpOp) const {
    Location loc = writeOp.getLoc();
    VectorType vecType = writeOp.getVectorType();

    // Only vector<1x> is supported at the moment.
    if (vecType.getShape().size() != 1 || vecType.getShape()[0] != 1)
      return failure();

    // Do not process warp ops that contain only TransferWriteOps.
    if (llvm::all_of(warpOp.getOps(), [](Operation &op) {
          return isa<vector::TransferWriteOp, vector_ext::YieldOp>(&op);
        }))
      return failure();

    SmallVector<Value> yieldValues = {writeOp.vector()};
    SmallVector<Type> retTypes = {vecType};
    WarpSingleLaneOp newWarpOp = moveRegionToNewWarpOpAndAppendReturns(
        rewriter, warpOp, yieldValues, retTypes);
    rewriter.setInsertionPointAfter(newWarpOp);

    // Create a second warp op that contains only writeOp.
    auto secondWarpOp = rewriter.create<WarpSingleLaneOp>(
        loc, TypeRange(), newWarpOp.laneid());
    Block &body = secondWarpOp.getBodyRegion().front();
    rewriter.setInsertionPointToStart(&body);
    auto newWriteOp =
        cast<vector::TransferWriteOp>(rewriter.clone(*writeOp.getOperation()));
    newWriteOp.vectorMutable().assign(
        newWarpOp.getResult(newWarpOp.getNumResults() - 1));
    rewriter.eraseOp(writeOp);
    rewriter.create<vector_ext::YieldOp>(newWarpOp.getLoc());
    return success();
  }

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    // Ops with mask not supported yet.
    if (writeOp.mask())
      return failure();

    auto warpOp = dyn_cast<WarpSingleLaneOp>(writeOp->getParentOp());
    if (!warpOp)
      return failure();

    // There must be no op with a side effect after writeOp.
    Operation *nextOp = writeOp.getOperation();
    while ((nextOp = nextOp->getNextNode()))
      if (hasSideEffect(*nextOp))
        return failure();

    if (!llvm::all_of(writeOp->getOperands(), [&](Value value) {
          return writeOp.vector() == value ||
                 warpOp.isDefinedOutsideOfRegion(value);
        }))
      return failure();

    if (succeeded(tryDistributeOp(rewriter, writeOp, warpOp)))
      return success();

    if (succeeded(tryExtractOp(rewriter, writeOp, warpOp)))
      return success();

    return failure();
  }

 private:
  DistributionMapFn distributionMapFn;
};

/// Sink scf.for region out of WarpSingleOp. This can be done only if the
/// scf.ForOp is the last operation in the region so that it doesn't change the
/// order of execution. This creates a new scf.for region after the
/// WarpSingleOp. The new scf.for region will contain a new WarpSingleLaneOp
/// region. Example:
/// ```
/// %w = vector_ext.warp_execute_on_lane_0(%laneid) -> (vector<4xf32>) {
///   ...
///   %v1 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %v)
///   -> (vector<128xf32>) {
///     ...
///     scf.yield %r : vector<128xf32>
///   }
///   vector_ext.yield %v1 : vector<128xf32>
/// }
/// ```
/// To:
/// %w0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<4xf32>) {
///   ...
///   vector_ext.yield %v : vector<128xf32>
/// }
/// %w = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%varg = %q0)
///   -> (vector<4xf32>) {
///     %iw = vector_ext.warp_execute_on_lane_0(%laneid)
///     args(%varg : vector<4xf32>) -> (vector<4xf32>) {
///     ^bb0(%arg: vector<128xf32>):
///       ...
///       vector_ext.yield %ir : vector<128xf32>
///     }
///     scf.yield %iw : vector<4xf32>
///  }
/// ```
struct WarpOpScfForOp : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    auto yield = cast<vector_ext::YieldOp>(
        warpOp.getBodyRegion().getBlocks().begin()->getTerminator());
    // Only pick up forOp if it is the last op in the region.
    Operation *lastNode = yield->getPrevNode();
    auto forOp = dyn_cast_or_null<scf::ForOp>(lastNode);
    if (!forOp)
      return failure();
    SmallVector<Value> newOperands;
    SmallVector<unsigned> resultIdx;
    // Collect all the outputs coming from the forOp.
    for (OpOperand &yieldOperand : yield->getOpOperands()) {
      if (yieldOperand.get().getDefiningOp() != forOp.getOperation())
        continue;
      auto forResult = yieldOperand.get().cast<OpResult>();
      newOperands.push_back(warpOp.getResult(yieldOperand.getOperandNumber()));
      yieldOperand.set(forOp.getIterOperands()[forResult.getResultNumber()]);
      resultIdx.push_back(yieldOperand.getOperandNumber());
    }
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(warpOp);
    // Create a new for op outside the region with a WarpSingleOp region inside.
    auto newForOp = rewriter.create<scf::ForOp>(
        forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newOperands);
    rewriter.setInsertionPoint(newForOp.getBody(), newForOp.getBody()->begin());
    auto innerWarp = rewriter.create<WarpSingleLaneOp>(
        warpOp.getLoc(), newForOp.getResultTypes(), warpOp.laneid(),
        newForOp.getRegionIterArgs(), forOp.getResultTypes());
    // Move the loop region within the new WarpSingleLaneOp region.
    BlockAndValueMapping mapping;
    mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
    for (auto args : llvm::zip(forOp.getRegionIterArgs(),
                               innerWarp.getBody()->getArguments())) {
      mapping.map(std::get<0>(args), std::get<1>(args));
    }
    rewriter.setInsertionPoint(innerWarp.getBody(),
                               innerWarp.getBody()->begin());
    for (Operation &innerOp : forOp.getBody()->without_terminator())
      rewriter.clone(innerOp, mapping);
    SmallVector<Value> yieldOperands;
    for (Value operand : forOp.getBody()->getTerminator()->getOperands())
      yieldOperands.push_back(mapping.lookup(operand));
    rewriter.create<vector_ext::YieldOp>(innerWarp.getLoc(), yieldOperands);
    rewriter.setInsertionPointAfter(innerWarp);
    rewriter.create<scf::YieldOp>(forOp.getLoc(), innerWarp.getResults());
    // remove the old forOp.
    forOp.getBody()->dropAllDefinedValueUses();
    rewriter.eraseOp(forOp);
    // Replace the warpOp result coming from the original ForOp.
    for (auto res : llvm::enumerate(resultIdx)) {
      warpOp.getResult(res.value())
          .replaceAllUsesWith(newForOp.getResult(res.index()));
      newForOp->setOperand(res.index() + 3, warpOp.getResult(res.value()));
    }
    return success();
  }
};

void mlir::vector_ext::populatePropagateVectorDistributionPatterns(
    RewritePatternSet &pattern) {
  pattern.add<WarpOpElementwise, WarpOpTransferRead, WarpOpDeadResult,
              WarpOpReduction, WarpOpBroadcast, WarpOpForwardOperand,
              WarpOpScfForOp>(pattern.getContext());
}

void mlir::vector_ext::populateDistributeTransferWriteOpPatterns(
    RewritePatternSet &patterns, DistributionMapFn distributionMapFn) {
  patterns.add<WarpOpTransferWrite>(patterns.getContext(), distributionMapFn);
}

static LogicalResult rewriteWarpOpToScfFor(
    RewriterBase &rewriter, WarpSingleLaneOp warpOp) {
  assert(warpOp.getBodyRegion().hasOneBlock() &&
         "expected WarpOp with single block");
  Block *warpOpBody = &warpOp.getBodyRegion().front();
  Location loc = warpOp.getLoc();

  // Passed all checks. Start rewriting.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(warpOp);

  // Create scf.if op.
  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value isLane0 = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 warpOp.laneid(), c0);
  auto ifOp = rewriter.create<scf::IfOp>(loc, isLane0,
                                         /*withElseRegion=*/false);
  rewriter.eraseOp(ifOp.thenBlock()->getTerminator());


  // Allocate a shared memory buffer for the given type.
  auto allocBuffer = [&](Type type) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(ifOp);
    if (auto vectorType = type.dyn_cast<VectorType>()) {
      return rewriter.create<memref::AllocOp>(loc,
          MemRefType::get(vectorType.getShape(), vectorType.getElementType()));
    } else {
      return rewriter.create<memref::AllocOp>(loc, MemRefType::get({1}, type));
    }
  };

  // Store vectors that are defined outside of warpOp into the scratch pad
  // buffer.
  SmallVector<Value> bbArgReplacements;
  for (const auto &it : llvm::enumerate(warpOp.args())) {
    Value val = it.value();
    Value bbArg = warpOpBody->getArgument(it.index());
    Value buffer = allocBuffer(bbArg.getType());

    // Store arg vector into buffer.
    rewriter.setInsertionPoint(ifOp);
    auto vectorType = val.getType().cast<VectorType>();
    int64_t storeSize = vectorType.getShape()[0];
    Value storeOffset = rewriter.create<arith::MulIOp>(loc, warpOp.laneid(),
        rewriter.create<arith::ConstantIndexOp>(loc, storeSize));
    rewriter.create<vector::StoreOp>(loc, val, buffer, storeOffset);

    // Load bbArg vector from buffer.
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    auto bbArgType = bbArg.getType().cast<VectorType>();
    Value loadOp = rewriter.create<vector::LoadOp>(loc, bbArgType, buffer, c0);
    bbArgReplacements.push_back(loadOp);
  }

  // Move body of warpOp to ifOp.
  rewriter.mergeBlocks(warpOpBody, ifOp.thenBlock(), bbArgReplacements);

  // Rewrite terminator and compute replacements of WarpOp results.
  SmallVector<Value> replacements;
  auto yieldOp = cast<vector_ext::YieldOp>(ifOp.thenBlock()->getTerminator());
  Location yieldLoc = yieldOp.getLoc();
  for (const auto &it : llvm::enumerate(yieldOp.operands())) {
    Value val = it.value();
    Type resultType = warpOp->getResultTypes()[it.index()];
    Value buffer = allocBuffer(val.getType());

    // Store yielded value into buffer.
    rewriter.setInsertionPoint(yieldOp);
    if (val.getType().isa<VectorType>())
      rewriter.create<vector::StoreOp>(yieldLoc, val, buffer, c0);
    else
      rewriter.create<memref::StoreOp>(yieldLoc, val, buffer, c0);

    // Load value from buffer (after warpOp).
    rewriter.setInsertionPointAfter(ifOp);
    if (resultType == val.getType()) {
      // Result type and yielded value type are the same. This is a broadcast.
      // E.g.:
      // %r = vector_ext.warp_execute_on_lane_0(...) -> (f32) {
      //   vector_ext.yield %cst : f32
      // }
      // Both types are f32. The constant %cst is broadcasted to all lanes.
      // This is described in more detail in the documentation of the op.
      Value loadOp = rewriter.create<memref::LoadOp>(loc, buffer, c0);
      replacements.push_back(loadOp);
    } else {
      auto loadedVectorType = resultType.cast<VectorType>();
      int64_t loadSize = loadedVectorType.getShape()[0];

      // loadOffset = laneid * loadSize
      Value loadOffset = rewriter.create<arith::MulIOp>(loc, warpOp.laneid(),
          rewriter.create<arith::ConstantIndexOp>(loc, loadSize));
      Value loadOp = rewriter.create<vector::LoadOp>(loc, loadedVectorType,
                                                     buffer, loadOffset);
      replacements.push_back(loadOp);
    }
  }

  // Delete terminator and add empty scf.yield.
  rewriter.eraseOp(yieldOp);
  rewriter.setInsertionPointToEnd(ifOp.thenBlock());
  rewriter.create<scf::YieldOp>(yieldLoc);

  // Compute replacements for WarpOp results.
  rewriter.replaceOp(warpOp, replacements);

  return success();
}

namespace {

struct WarpOpToScfForPattern : public OpRewritePattern<WarpSingleLaneOp> {
  using OpRewritePattern<WarpSingleLaneOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(WarpSingleLaneOp warpOp,
                                PatternRewriter &rewriter) const override {
    return rewriteWarpOpToScfFor(rewriter, warpOp);
  }
};

} // namespace

void mlir::vector_ext::populateWarpSingleLaneOpToScfForPattern(
    RewritePatternSet &patterns) {
  patterns.add<WarpOpToScfForPattern>(patterns.getContext());
}

/// Helper to know if an op can be hoisted out of the region.
static bool canBeHoisted(Operation *op,
                         function_ref<bool(Value)> definedOutside) {
  return llvm::all_of(op->getOperands(), definedOutside) &&
         !hasSideEffect(*op) && op->getNumRegions() == 0;
}

void mlir::vector_ext::moveScalarUniformCode(WarpSingleLaneOp warpOp) {
  Block *body = warpOp.getBody();

  // Keep track of the ops we want to hoist.
  llvm::SmallSetVector<Operation *, 8> opsToMove;

  // Helper to check if a value is or will be defined outside of the region.
  auto isDefinedOutsideOfBody = [&](Value value) {
    auto *definingOp = value.getDefiningOp();
    return (definingOp && opsToMove.count(definingOp)) ||
           warpOp.isDefinedOutsideOfRegion(value);
  };

  // Do not use walk here, as we do not want to go into nested regions and hoist
  // operations from there.
  for (auto &op : body->without_terminator()) {
    bool hasVectorResult = llvm::any_of(op.getResults(), [](Value result) {
      return result.getType().isa<VectorType>();
    });
    if (!hasVectorResult && canBeHoisted(&op, isDefinedOutsideOfBody))
      opsToMove.insert(&op);
  }

  // Move all the ops marked as uniform outside of the region.
  for (Operation *op : opsToMove)
    op->moveBefore(warpOp);
}
