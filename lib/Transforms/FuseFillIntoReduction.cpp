//===- FuseFillIntoReduction.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Transforms/Transforms.h"
#include "mlir/Dialect/SCF/SCF.h"

namespace mlir {
namespace linalg {
namespace {

using mlir::tensor::ExtractSliceOp;
using mlir::tensor::InsertSliceOp;

// Tiles a GenericOp that models a reduction and then fuses its inputs and
// outputs. Currently, only the FillOp that initializes the output is fused into
// the loop nest.
struct FuseFillOutputIntoGenericOpPattern
    : public OpInterfaceRewritePattern<LinalgOp> {
  FuseFillOutputIntoGenericOpPattern(LinalgTilingOptions options,
                                     LinalgTransformationFilter filter,
                                     MLIRContext *context,
                                     PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<LinalgOp>(context, benefit),
        filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

    auto tiledOp = tileLinalgOp(rewriter, linalgOp, options);
    if (failed(tiledOp)) return failure();

    auto outerLoop = dyn_cast<scf::ForOp>(tiledOp->loops.front());
    if (!outerLoop) return failure();

    if (failed(FuseFillOp(rewriter, outerLoop, tiledOp->op))) {
      return failure();
    }
    rewriter.replaceOp(linalgOp, outerLoop->getResults());

    outerLoop->walk([&](LinalgOp tiledOp) {
      filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
    });
    return success();
  }

 private:
  // Replaces
  //
  // %0 = linalg.fill(%cst, %out)
  //
  // with
  //
  // %0 = linalg.fill(%cst, %out)
  // %1 = linalg.fill(%cst, %0)
  //
  // The idea is to still initialize the output of the reduction even if the
  // FillOp is fused into the loop nest. In that case %1 will be fused into
  // the loop body and %0 will remain outside of the loop.
  std::pair<FillOp, FillOp> ChainFillOp(PatternRewriter &rewriter,
                                        FillOp fillOp) const {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(fillOp);

    auto *first = rewriter.clone(*fillOp);
    auto second = rewriter.replaceOpWithNewOp<FillOp>(fillOp, fillOp.value(),
                                                      first->getResult(0));
    return std::make_pair(cast<FillOp>(first), second);
  }

  // Fuses FillOp producer of the output argument of the loop nest and inserts
  // an operation that accumulates the partial result, i.e. reduced tile, and
  // the current value of the output tile.
  LogicalResult FuseFillOp(PatternRewriter &rewriter, scf::ForOp outerLoop,
                           LinalgOp tiledOp) const {
    if (tiledOp.getNumOutputs() != 1) return failure();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(tiledOp);

    if (outerLoop.getNumIterOperands() == 0) return failure();
    auto fillOp = outerLoop.getIterOperands().front().getDefiningOp<FillOp>();
    if (!fillOp) return failure();

    auto fillOpChain = ChainFillOp(rewriter, fillOp);

    Optional<linalg::FusionInfo> fusionInfo =
        linalg::fuseProducerOfTensor(rewriter, fillOpChain.second->getResult(0),
                                     *tiledOp.getOutputOperands().front());
    if (!fusionInfo.hasValue()) return failure();

    rewriter.replaceOp(fillOpChain.second, fillOpChain.first.getResult(0));

    auto fusedFillOp = cast<FillOp>(fusionInfo->fusedProducer);

    Value partialResult = tiledOp->getResult(0);

    // Find insert_slice that inserts the result back to the output.
    auto insert = dyn_cast<InsertSliceOp>(*partialResult.getUsers().begin());
    if (!insert) return failure();

    // Create operation that accumulates the partial result into the output.
    auto numParallelLoops = tiledOp.getNumParallelLoops();
    SmallVector<StringRef, 3> parallelIterTypes(numParallelLoops,
                                                getParallelIteratorTypeName());
    auto idMap = rewriter.getMultiDimIdentityMap(numParallelLoops);

    auto loc = tiledOp.getLoc();
    auto accumulator = rewriter.create<GenericOp>(
        loc, partialResult.getType(), llvm::makeArrayRef(partialResult),
        llvm::makeArrayRef(fusedFillOp.output()),
        llvm::makeArrayRef({idMap, idMap}), parallelIterTypes);

    auto reduceTile = cast<GenericOp>(tiledOp);
    BlockAndValueMapping bvm;
    rewriter.cloneRegionBefore(reduceTile.region(), accumulator.region(),
                               accumulator.region().end(), bvm);
    rewriter.updateRootInPlace(insert, [&]() {
      insert.sourceMutable().assign(accumulator.getResult(0));
    });
    return success();
  }

  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};

// Rewrite linalg.fill(extract_slice) as linalg.fill(init_tensor). This rewrite
// is required for correctness, because otherwise after bufferization the fused
// output linalg.fill would still use the buffer for the reduction of the whole
// output instead of allocating a local buffer only for the reduced tile.
//
// A better way to perform this transformation is to have it in MLIR Core as a
// part of the fusion logic. To support this correctly, we would also modify
// logic for padding, so that we could pad fill(init_tensor). Currently, only
// fill(extract_slice) can be padded. All these changes will happen once we
// converge on the pipeline design.
struct FillOfExtractSlice : public OpRewritePattern<FillOp> {
  using OpRewritePattern<FillOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FillOp fill,
                                PatternRewriter &rewriter) const override {
    if (!fill.hasTensorSemantics()) return failure();

    auto fillTensorType = fill.getOutputTensorTypes().back();
    if (!fillTensorType.hasStaticShape()) return failure();

    if (auto extract = fill.output().getDefiningOp<ExtractSliceOp>()) {
      llvm::SmallVector<int64_t, 4> staticSizes = llvm::to_vector<4>(
          llvm::map_range(extract.static_sizes().cast<ArrayAttr>(),
                          [](Attribute a) -> int64_t {
                            return a.cast<IntegerAttr>().getInt();
                          }));
      auto init = rewriter.create<linalg::InitTensorOp>(
          fill.getLoc(), extract.getDynamicSizes(), staticSizes,
          fillTensorType.getElementType());
      rewriter.replaceOpWithNewOp<FillOp>(fill, fill.value(), init);
      return success();
    }
    return failure();
  }
};

// Match 2D row reduction. This is a starting point, we will relax this
// condition further down the road, when we add support for more reduction
// types.
bool is2DRowReduction(Operation *op) {
  auto reduction = dyn_cast<GenericOp>(op);
  if (!reduction) return false;

  if (reduction.getNumOutputs() != 1 || reduction.getNumLoops() != 2)
    return false;
  return reduction.getNumReductionLoops() == 1;
}

}  // namespace

void populateFuseFillIntoReductionPatterns(OwningRewritePatternList &patterns,
                                           const LinalgTilingOptions &opts) {
  auto *ctx = patterns.getContext();
  auto filter =
      LinalgTransformationFilter(llvm::None, {Identifier::get("tiled", ctx)})
          .addFilter(
              [](Operation *op) { return success(is2DRowReduction(op)); });
  patterns.insert<linalg::FuseFillOutputIntoGenericOpPattern>(opts, filter,
                                                              ctx);
  patterns.insert<FillOfExtractSlice>(patterns.getContext());
}

}  // namespace linalg
}  // namespace mlir
