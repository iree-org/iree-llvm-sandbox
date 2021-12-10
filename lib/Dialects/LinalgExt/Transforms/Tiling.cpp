//===- Tiling.cpp - Tiling using TilingInterface --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "Dialects/LinalgExt/PassDetail.h"
#include "Dialects/LinalgExt/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <mlir/IR/OperationSupport.h>

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

SmallVector<Value> tileToSCF(PatternRewriter &rewriter, TilingInterface op,
                             TilingInterface clonedOp, ValueRange tileSizes) {
  // Compute lower and upper bounds of the loop nest.
  SmallVector<Range> ranges = clonedOp.getIterationDomain(rewriter);
  assert(tileSizes.size() == ranges.size() &&
         "expected tile sizes to match the number of loops");
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
  llvm::SmallPtrSet<Operation *, 1> preservedUses;
  SmallVector<Value> destOperand = clonedOp.getDestinationOperands(rewriter);
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
        assert(clonedOp->getNumResults() == 1 && "expected single result op");
        Value tiledOutput =
            linalg::makeTiledShape(b, loc, clonedOp->getResult(0), tileSizes,
                                   map, offsets, allDims, sizes);
        auto sliceOp = tiledOutput.getDefiningOp<tensor::ExtractSliceOp>();
        preservedUses.insert(sliceOp);
        assert(sliceOp && "expected ExtractSliceOp");
        // Insert the tile into the output tensor.
        Value yieldValue =
            insertSliceIntoTensor(b, loc, sliceOp, sliceOp, iterArgs[0]);
        return scf::ValueVector({yieldValue});
      });
  return loopNest.getResults();
}

namespace {

/// The tiling here works by two steps. The first step is to create a loop based
/// on the loop bounds of the operation obtained from `TilingInterface`.
///
/// ```mlir
///   %1 = <tiling interface op> ins(...) outs(%0 : ...)
///   ... <use_op> ... %1 ...
/// ```
///
/// is rewritten using a "noop" subtensor extract/insert pair
///
/// ```mlir
///   %1 = <tiling interface op> ins(...) outs(%0 : ...)
///   %2 = scf.for %iv0 = ... iter_args(%arg0 = %0) {
///     %3 = scf.for %iv1 = ... iter_args(%arg1 = %arg0) {
///       ...
///       %4 = tensor.extract_slice %1[%iv0, %iv1]....
///       %5 = tensor.insert_slice %4 into %arg1[%iv0, %iv1]...
///       scf.yield %5
///     }
///     scf.yield %3
///   }
///   ... <use_op> ... %2 ...
/// ```
///
/// Following this the `TilingInterface` -> `tensor::ExtractSliceOp` pattern is
/// replaced with
///
/// /// ```mlir
///   %2 = scf.for %iv0 = ... iter_args(%arg0 = %0) {
///     %3 = scf.for %iv1 = ... iter_args(%arg1 = %arg0) {
///       ...
///       %4 = tensor.extract_slice %0[%iv0, %iv1]
///       %5 = <tiling interface op> ins(...) outs(%4 : ...)
///       %6 = tensor.insert_slice %5 into %arg1[%iv0, %iv1]...
///       scf.yield %6
///     }
///     scf.yield %3
///   }
///   ... <use_op> ... %2 ...
/// ```
///
/// TODO(ravishankarm): The current approach seems to work for only tiling the
/// parallel loops of the operation. Specifically,
/// 1) the `%0` in the third snippet needs to be `%arg1`, for cases where the
///    tiled loop is a reduction.
/// 2) Current implementation is using the `getIterationDomain` method to get
/// the
///    initial loop structure as described in the second snippet. If any of
///    those loops are reductions, then that IR snippet itself is wrong (replace
///    this with the case of `linalg.matmul` and the error becomes apparent).

/// First pattern to introduce the loop nests.
struct OpTilingPattern : public OpInterfaceRewritePattern<TilingInterface> {
  OpTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions opt,
                  linalg::LinalgTransformationFilter filt)
      : OpInterfaceRewritePattern<TilingInterface>(context), options(opt),
        filter(filt) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();

    /// Currently only handle single result operations.
    if (op->getNumResults() != 1)
      return failure();

    /// Currently only handle operations with all parallel iterator types.
    if (llvm::any_of(op.getLoopIteratorTypes(), [](StringRef iteratorType) {
          return iteratorType != getParallelIteratorTypeName();
        })) {
      return failure();
    }

    // Get rank and tile sizes.
    SmallVector<Value> tileSizes =
        options.tileSizeComputationFunction(rewriter, op);
    auto clonedOp = cast<TilingInterface>(rewriter.clone(*op.getOperation()));
    SmallVector<Value> results = tileToSCF(rewriter, op, clonedOp, tileSizes);

    filter.replaceLinalgTransformationFilter(rewriter, clonedOp);
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  linalg::LinalgTilingOptions options;
  linalg::LinalgTransformationFilter filter;
};

/// Second pattern to implement the switch of `TilingInterface ->
/// tensor.extract_slice` to `tensor.extract_slice -> `TilingInterface`.
struct SliceOpTiledOpSwapPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  SliceOpTiledOpSwapPattern(MLIRContext *context,
                            linalg::LinalgTilingOptions opt,
                            linalg::LinalgTransformationFilter filt)
      : OpRewritePattern<tensor::ExtractSliceOp>(context), options(opt),
        filter(filt) {}

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp sliceOp,
                                PatternRewriter &rewriter) const override {
    auto sourceOp = sliceOp.source().getDefiningOp<TilingInterface>();
    if (!sourceOp || !filter.hasReplacementFilter(sourceOp))
      return failure();
    SmallVector<Operation *> tiledOps = sourceOp.getTiledImplementation(
        rewriter, sourceOp.getDestinationOperands(rewriter),
        sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
        /*tileDestOperands=*/true);
    assert(tiledOps.size() && "expected single tiled op");
    Operation *tiledOp = tiledOps.front();
    rewriter.replaceOp(sliceOp, tiledOp->getResults());
    return success();
  }

private:
  linalg::LinalgTilingOptions options;
  linalg::LinalgTransformationFilter filter;
};

/// Pass to test the tiling tranforamtion.
struct LinalgExtTilingPass : public LinalgExtTilingBase<LinalgExtTilingPass> {
  LinalgExtTilingPass() = default;
  LinalgExtTilingPass(ArrayRef<int64_t> tileSizes) {
    this->tileSizes = tileSizes;
  }
  void runOnOperation() override;
};
} // namespace

void LinalgExtTilingPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();

  RewritePatternSet patterns(context);

  auto options = linalg::LinalgTilingOptions().setTileSizes(tileSizes);
  auto filter = linalg::LinalgTransformationFilter(
      ArrayRef<Identifier>{}, Identifier::get("tiled", context));
  patterns.insert<OpTilingPattern, SliceOpTiledOpSwapPattern>(context, options,
                                                              filter);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::linalg_ext::createLinalgExtTilingPass(ArrayRef<int64_t> tileSizes) {
  return std::make_unique<LinalgExtTilingPass>(tileSizes);
}
