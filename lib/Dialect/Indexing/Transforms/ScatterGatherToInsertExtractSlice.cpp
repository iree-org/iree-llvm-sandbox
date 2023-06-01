//===-- ScatterGatherToInsertExtractSlice.cpp - Pass Implementation -------*-
// C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Indexing/Transforms/ScatterGatherToInsertExtractSlice.h"
#include "structured/Dialect/Indexing/IR/Indexing.h"
#include "structured/Dialect/Indexing/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <tuple>

namespace mlir {
#define GEN_PASS_DEF_GATHERTOEXTRACTSLICE
#define GEN_PASS_DEF_SCATTERTOINSERTSLICE
#include "structured/Dialect/Indexing/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::indexing;

namespace {

SmallVector<OpFoldResult, 3>
getMakeOffsetSizeStrideFromARange(ARangeOp arangeOp,
                                  PatternRewriter &rewriter) {
  SmallVector<OpFoldResult, 3> results;
  SmallVector<StringAttr> attrs = {arangeOp.getStartAttrAttrName(),
                                   arangeOp.getStopAttrAttrName(),
                                   arangeOp.getStepAttrAttrName()};
  SmallVector<Value> opers = {arangeOp.getStart(), arangeOp.getStop(),
                              arangeOp.getStep()};
  for (const auto &[index, tuple] : llvm::enumerate(llvm::zip(opers, attrs))) {
    auto val = std::get<0>(tuple);
    auto attrName = std::get<1>(tuple);
    if (val)
      results.push_back(val);
    else
      results.push_back(arangeOp->getAttr(attrName));
  }

  if (results[0].is<Attribute>() && results[1].is<Attribute>() &&
      results[2].is<Attribute>())
    results[1] =
        rewriter.getIndexAttr((getConstantIntValue(results[1]).value() -
                               getConstantIntValue(results[0]).value()) /
                              getConstantIntValue(results[2]).value());
  else {

    auto start = results[0].is<Value>()
                     ? results[0].dyn_cast<Value>()
                     : rewriter.create<arith::ConstantOp>(
                           arangeOp->getLoc(),
                           rewriter.getSI32IntegerAttr(
                               getConstantIntValue(results[0]).value()));
    auto stop = results[1].is<Value>()
                    ? results[1].dyn_cast<Value>()
                    : rewriter.create<arith::ConstantOp>(
                          arangeOp->getLoc(),
                          rewriter.getSI32IntegerAttr(
                              getConstantIntValue(results[1]).value()));
    auto step = results[2].is<Value>()
                    ? results[2].dyn_cast<Value>()
                    : rewriter.create<arith::ConstantOp>(
                          arangeOp->getLoc(),
                          rewriter.getSI32IntegerAttr(
                              getConstantIntValue(results[2]).value()));
    auto diff = rewriter.create<arith::SubIOp>(arangeOp->getLoc(), stop, start)
                    .getResult();
    results[1] =
        rewriter.create<arith::FloorDivSIOp>(arangeOp->getLoc(), diff, step)
            .getResult();
  }

  return results;
}

SmallVector<SmallVector<OpFoldResult, 3>, 3> getMakeMixedOffsetsSizesStrides(
    const Value indices, ArrayRef<int64_t> collapseDims,
    ArrayRef<int64_t> sourceDestShape, PatternRewriter &rewriter) {

  SmallVector<ARangeOp, 3> arangeOps;
  if (auto arangeOp = indices.getDefiningOp<indexing::ARangeOp>()) {
    arangeOps.push_back(arangeOp);
  } else {
    for (const auto &arangeVal :
         indices.getDefiningOp<indexing::MeshGridOp>()->getOperands())
      arangeOps.push_back(arangeVal.getDefiningOp<ARangeOp>());
  }

  llvm::SmallMapVector<int64_t, int64_t, 4> collapseDimsMap;
  for (const auto &item : llvm::enumerate(collapseDims))
    collapseDimsMap[item.value()] = item.index();

  SmallVector<OpFoldResult, 3> mixedOffsets;
  SmallVector<OpFoldResult, 3> mixedSizes;
  SmallVector<OpFoldResult, 3> mixedStrides;
  for (int dim = 0; dim < sourceDestShape.size(); ++dim) {
    if (collapseDimsMap.contains(dim)) {
      auto arangeOp = arangeOps[collapseDimsMap[dim]];
      auto offsetSizeStride =
          getMakeOffsetSizeStrideFromARange(arangeOp, rewriter);
      mixedOffsets.push_back(offsetSizeStride[0]);
      mixedSizes.push_back(offsetSizeStride[1]);
      mixedStrides.push_back(offsetSizeStride[2]);
    } else {
      mixedOffsets.push_back(rewriter.getI64IntegerAttr(0));
      mixedSizes.push_back(rewriter.getI64IntegerAttr(sourceDestShape[dim]));
      mixedStrides.push_back(rewriter.getI64IntegerAttr(1));
    }
  }
  return {mixedOffsets, mixedSizes, mixedStrides};
}

struct ConvertScatterOpToInsertSliceOp : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;
  LogicalResult match(ScatterOp op) const override {
    if (op.getSource().getType().hasRank() &&
        op.getDest().getType().hasRank() && isARangeIndices(op.getIndices())) {
      return success();
    }
    return failure();
  }

  void rewrite(ScatterOp op, PatternRewriter &rewriter) const override {
    auto mixedOffsetsSizesStrides = getMakeMixedOffsetsSizesStrides(
        op.getIndices(), op.getScatterDims(), op.getDest().getType().getShape(),
        rewriter);
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        op, op.getSource(), op.getDest(), mixedOffsetsSizesStrides[0],
        mixedOffsetsSizesStrides[1], mixedOffsetsSizesStrides[2]);
  }
};

struct ConvertGatherOpToInsertSliceOp : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;
  LogicalResult match(GatherOp op) const override {
    if (op.getSource().getType().hasRank() &&
        isARangeIndices(op.getIndices())) {
      return success();
    }
    return failure();
  }

  void rewrite(GatherOp op, PatternRewriter &rewriter) const override {
    auto mixedOffsetsSizesStrides = getMakeMixedOffsetsSizesStrides(
        op.getIndices(), op.getGatherDims(),
        op.getSource().getType().getShape(), rewriter);
    auto resultType = tensor::ExtractSliceOp::inferResultType(
        op.getSource().getType(), mixedOffsetsSizesStrides[0],
        mixedOffsetsSizesStrides[1], mixedOffsetsSizesStrides[2]);
    auto newOp = rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, resultType, op.getSource(), mixedOffsetsSizesStrides[0],
        mixedOffsetsSizesStrides[1], mixedOffsetsSizesStrides[2]);
  }
};

} // namespace

namespace {

struct GatherToExtractSlicePass
    : public mlir::impl::GatherToExtractSliceBase<GatherToExtractSlicePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertGatherOpToInsertSliceOp>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  };
};

struct ScatterToInsertSlicePass
    : public mlir::impl::ScatterToInsertSliceBase<ScatterToInsertSlicePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ConvertScatterOpToInsertSliceOp>(&getContext());
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  };
};

} // namespace

std::unique_ptr<Pass> mlir::createGatherToExtractSlicePass() {
  return std::make_unique<GatherToExtractSlicePass>();
}

std::unique_ptr<Pass> mlir::createScatterToInsertSlicePass() {
  return std::make_unique<ScatterToInsertSlicePass>();
}
