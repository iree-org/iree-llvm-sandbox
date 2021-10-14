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

namespace mlir {
namespace linalg {

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

}  // namespace linalg
}  // namespace mlir

static Optional<TiledLoopOp> tileAndFuseLinalgOp(
    PatternRewriter &rewriter, LinalgOp linalgOp,
    const LinalgTilingOptions &tilingOptions) {
  auto tiledLinalgOp = tileLinalgOp(rewriter, linalgOp, tilingOptions);
  if (!tiledLinalgOp) return llvm::None;

  linalg::fuseProducerOfTensor(rewriter,
                               linalgOp.getOutputOperands()
                                   .front()
                                   ->get()
                                   .getDefiningOp()
                                   ->getResults()
                                   .front(),
                               *tiledLinalgOp->op.getOutputOperands().front());

  // Try to pad on the fly by rewriting tiledLinalgOp->op as a padded op.
  // TODO: This requires padding and bounding box to symbolic multiples.
  // (void)rewriteAsPaddedOp(rewriter, *tiledLinalgOp, tilingOptions);

  return tiledLinalgOp.getValue().op->getParentOfType<TiledLoopOp>();
}

LogicalResult mlir::linalg::TileAndFusePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

  Optional<TiledLoopOp> tiledLoopOp =
      tileAndFuseLinalgOp(rewriter, op, options);
  if (!tiledLoopOp) return failure();
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
  patterns.insert<mlir::linalg::TileAndFusePattern>(opts, filter,
                                                    patterns.getContext());
}
