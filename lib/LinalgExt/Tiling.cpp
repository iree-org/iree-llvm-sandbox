//===- Tiling.cpp - Tiling using TilingInterface --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LinalgExt/LinalgExtOps.h"
#include "LinalgExt/PassDetail.h"
#include "LinalgExt/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg_ext;

static bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  return false;
}

// Insert a tile `source` into the destination tensor `dest`. The position at
// which the tile is inserted (as well as size of tile) is taken from a given
// ExtractSliceOp `sliceOp`.
static Value insertSliceIntoTensor(OpBuilder &b, Location loc,
                                   tensor::ExtractSliceOp sliceOp, Value source,
                                   Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, sliceOp.source().getType(), source, dest, sliceOp.offsets(),
      sliceOp.sizes(), sliceOp.strides(), sliceOp.static_offsets(),
      sliceOp.static_sizes(), sliceOp.static_strides());
}

namespace {

template <typename TiledOp>
struct OpTilingPattern : public OpRewritePattern<TiledOp> {
  OpTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions opt,
                  linalg::LinalgTransformationFilter filt)
      : OpRewritePattern<TiledOp>(context), options(opt), filter(filt) {}

  LogicalResult matchAndRewrite(TiledOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op))) return failure();

    // Clone operation so that existing op can be replaced easily.
    auto clonedOp = cast<TiledOp>(rewriter.clone(*op.getOperation()));

    // Get rank and tile sizes.
    SmallVector<Value> tileSizes =
        options.tileSizeComputationFunction(rewriter, op);
    // Compute lower and upper bounds of the loop nest.
    SmallVector<Range> ranges = op.getLoopBounds(rewriter);
    assert(static_cast<int64_t>(tileSizes.size()) == ranges.size());
    SmallVector<Value> lbs, dims, allDims, steps;
    for (auto it : llvm::enumerate(ranges)) {
      allDims.push_back(it.value().size);
      if (!isZero(tileSizes[it.index()])) {
        lbs.push_back(it.value().offset);
        dims.push_back(it.value().size);
        steps.push_back(tileSizes[it.index()]);
      }
    }

    // Generate loop nest: One loop per dimension.
    SmallVector<Value> destOperand = op.getDestinationOperands(rewriter);
    Location loc = op->getLoc();
    auto loopNest = mlir::scf::buildLoopNest(
        rewriter, loc, lbs, /*ubs=*/dims, steps, ValueRange(destOperand),
        [&](OpBuilder &b, Location loc, ValueRange localIvs,
            ValueRange iterArgs) -> scf::ValueVector {
          // Compute offsets and sizes of ExtractSliceOp.
          SmallVector<Value> offsets =
              linalg::computeTileOffsets(b, loc, localIvs, tileSizes);
          SmallVector<Value> sizes =
              linalg::computeTileSizes(b, loc, localIvs, tileSizes, allDims);
          // Create ExtractSliceOp: Extract a tile from the PadTensorOp.
          // Note: The PadTensorOp is located outside of the loop nest. It is
          // later moved inside by ExtractSliceOfPadTensorSwapPattern.
          auto map =
              AffineMap::getMultiDimIdentityMap(ranges.size(), b.getContext());
          Value tiledOutput =
              linalg::makeTiledShape(b, loc, clonedOp->getResult(0), tileSizes,
                                     map, offsets, allDims, sizes);
          auto sliceOp = tiledOutput.getDefiningOp<tensor::ExtractSliceOp>();
          assert(sliceOp && "expected ExtractSliceOp");
          // Insert the tile into the output tensor.
          Value yieldValue =
              insertSliceIntoTensor(b, loc, sliceOp, sliceOp, iterArgs[0]);
          return scf::ValueVector({yieldValue});
        });

    filter.replaceLinalgTransformationFilter(rewriter, op);
    rewriter.replaceOp(op, loopNest.getResults());
    return success();
  }

 private:
  linalg::LinalgTilingOptions options;
  linalg::LinalgTransformationFilter filter;
};

template <typename TiledOp>
struct SliceOpTiledOpSwapPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  SliceOpTiledOpSwapPattern(MLIRContext *context,
                            linalg::LinalgTilingOptions opt,
                            linalg::LinalgTransformationFilter filt)
      : OpRewritePattern<tensor::ExtractSliceOp>(context),
        options(opt),
        filter(filt) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto sourceOp = sliceOp.source().getDefiningOp<TiledOp>();
    if (!sourceOp) return failure();
    if (failed(filter.checkAndNotify(rewriter, sourceOp))) return failure();
    Operation *tiledOp = sourceOp.getTiledImplementation(
        rewriter, sourceOp.outputs(), sliceOp.getMixedOffsets(),
        sliceOp.getMixedSizes());
    rewriter.replaceOp(sliceOp, tiledOp->getResults());
    filter.replaceLinalgTransformationFilter(rewriter, sourceOp);
    filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    return success();
  }

 private:
  linalg::LinalgTilingOptions options;
  linalg::LinalgTransformationFilter filter;
};

struct LinalgExtTilingPass : public LinalgExtTilingBase<LinalgExtTilingPass> {
  LinalgExtTilingPass() = default;
  LinalgExtTilingPass(ArrayRef<int64_t> tileSizes) {
    this->tileSizes = tileSizes;
  }
  void runOnOperation() override;
};
}  // namespace

void LinalgExtTilingPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();

  RewritePatternSet patterns(context);

  auto options = linalg::LinalgTilingOptions().setTileSizes(tileSizes);
  auto filter = linalg::LinalgTransformationFilter(
      ArrayRef<Identifier>{}, Identifier::get("tiled", context));
  patterns.insert<OpTilingPattern<linalg_ext::ReverseOp>,
                  SliceOpTiledOpSwapPattern<linalg_ext::ReverseOp>>(
      context, options, filter);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::linalg_ext::createLinalgExtTilingPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<LinalgExtTilingPass>(tileSizes);
}
