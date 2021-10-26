//===- LinalgTileToGenericPass.cpp - Tile and distribute to linalg.tile ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for tile and distribute to a tiled nested linalg
// abstraction.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"

#define DEBUG_TYPE "tile-fuse"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct TileAndFusePattern : public RewritePattern {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  TileAndFusePattern(LinalgTilingOptions options,
                     LinalgTransformationFilter filter,
                     mlir::MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(filter),
        options(options) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

 private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  LinalgTilingOptions options;
};
}  // namespace

static FailureOr<TiledLoopOp> tileAndFuseLinalgOp(
    PatternRewriter &rewriter, LinalgOp linalgOp,
    const LinalgTilingOptions &tilingOptions) {
  auto tiledLinalgOp = tileLinalgOp(rewriter, linalgOp, tilingOptions);
  if (failed(tiledLinalgOp)) return failure();

  if (failed(linalg::fuseProducerOfTensor(
          rewriter,
          linalgOp.getOutputOperands()
              .front()
              ->get()
              .getDefiningOp()
              ->getResults()
              .front(),
          *tiledLinalgOp->op.getOutputOperands().front())))
    return failure();

  // Try to pad on the fly by rewriting tiledLinalgOp->op as a padded op.
  // TODO: This requires padding and bounding box to symbolic multiples.
  // (void)rewriteAsPaddedOp(rewriter, *tiledLinalgOp, tilingOptions);

  if (auto res = tiledLinalgOp.getValue().op->getParentOfType<TiledLoopOp>())
    return res;
  return failure();
}

LogicalResult TileAndFusePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

  auto tiledLoopOp = tileAndFuseLinalgOp(rewriter, op, options);

  if (failed(tiledLoopOp)) return failure();
  if (tiledLoopOp->getNumResults() > 0)
    rewriter.replaceOp(op, tiledLoopOp->getResults());
  else
    rewriter.eraseOp(op);
  tiledLoopOp->walk([&](LinalgOp tiledOp) {
    filter.replaceLinalgTransformationFilter(rewriter, tiledOp);
  });
  return success();
}

void mlir::linalg::populateTileAndFusePattern(
    OwningRewritePatternList &patterns, const LinalgTilingOptions &opts,
    const LinalgTransformationFilter &filter) {
  patterns.insert<TileAndFusePattern>(opts, filter, patterns.getContext());
}
