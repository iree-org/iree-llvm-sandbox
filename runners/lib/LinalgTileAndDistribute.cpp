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

#define DEBUG_TYPE "distribute"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

namespace mlir {
namespace linalg {

struct TileAndDistributePattern : public RewritePattern {
  /// MatchAnyOpTag-based constructor with a mandatory `filter`.
  TileAndDistributePattern(TileAndDistributeOptions options,
                           LinalgTransformationFilter filter,
                           mlir::MLIRContext *context,
                           PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context),
        filter(filter),
        options(options) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;

 private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  TileAndDistributeOptions options;
};

}  // namespace linalg
}  // namespace mlir

static TiledLoopOp buildTiledLoop(PatternRewriter &rewriter,
                                  TiledLinalgOp &&tiledLinalgOp) {
  auto outerLoop = cast<scf::ForOp>(tiledLinalgOp.loops.front());
  auto innerLoop = cast<scf::ForOp>(tiledLinalgOp.loops.back());

  Location loc = tiledLinalgOp.op->getLoc();

  // Collect bounds and steps.
  SmallVector<Value> lbs, ubs, steps, ivs;
  for (Operation *loop : tiledLinalgOp.loops) {
    scf::ForOp forOp = cast<scf::ForOp>(loop);
    lbs.push_back(forOp.lowerBound());
    ubs.push_back(forOp.upperBound());
    steps.push_back(forOp.step());
    ivs.push_back(forOp.getInductionVar());
  }

  // Collect outputs.
  SmallVector<Value> outputs{outerLoop.getIterOperands()};
  for (auto &out : outputs) {
    auto defOp = out.getDefiningOp<linalg::FillOp>();
    if (defOp) out = defOp.output();
  }

  // Collect inputs.
  SmallVector<Value> inputs;
  for (SubTensorOp sub : innerLoop.getOps<SubTensorOp>()) {
    if (sub.getResult().getUses().empty()) continue;
    auto src = sub.source();

    if (llvm::is_contained(outputs, src)) continue;
    if (auto opResult = src.dyn_cast<OpResult>()) {
      if (!outerLoop->isProperAncestor(opResult.getDefiningOp())) {
        inputs.push_back(src);
        continue;
      }
    }
    if (auto bbArg = src.dyn_cast<BlockArgument>()) {
      if (!outerLoop->isAncestor(bbArg.getOwner()->getParentOp())) {
        inputs.push_back(src);
        continue;
      }
    }
  }

  SmallVector<StringRef> iterTypes = {"parallel"};
  auto tiledLoop =
      rewriter.create<TiledLoopOp>(loc, lbs, ubs, steps, inputs, outputs,
                                   rewriter.getStrArrayAttr(iterTypes));
  // Move the ops.
  auto loopBuilder =
      OpBuilder::atBlockBegin(tiledLoop.getBody(), rewriter.getListener());
  BlockAndValueMapping map;
  map.map(ivs, tiledLoop.getInductionVars());
  map.map(innerLoop.getRegionIterArgs(), outputs);

  for (auto &op : innerLoop.getBody()->without_terminator())
    loopBuilder.clone(op, map);

  // Convert the terminator
  SmallVector<Value> results;
  for (auto result : innerLoop.getBody()->getTerminator()->getOperands())
    results.push_back(map.lookup(result));
  loopBuilder.create<linalg::YieldOp>(loc, results);

  rewriter.replaceOp(outerLoop, tiledLoop.getResults());

  return tiledLoop;
}

static Optional<TiledLoopOp> tileAndDistributeLinalgOp(
    PatternRewriter &rewriter, LinalgOp linalgOp,
    const TileAndDistributeOptions &options) {
  auto tiledLinalgOp = tileLinalgOp(rewriter, linalgOp, options.tilingOptions);
  if (!tiledLinalgOp) return llvm::None;
  linalg::fuseProducerOfTensor(rewriter,
                               linalgOp.getOutputOpOperands()
                                   .front()
                                   .get()
                                   .getDefiningOp()
                                   ->getResults()
                                   .front(),
                               tiledLinalgOp->op.getOutputOpOperands().front());

  // Consider padding on the fly only if the op has tensor semantics.
  if (!options.tilingOptions.paddingValueComputationFunction ||
      !linalgOp.hasTensorSemantics())
    return buildTiledLoop(rewriter, std::move(*tiledLinalgOp));

  // Try to pad on the fly by rewriting tiledLinalgOp->op as a padded op.
  // TODO: This requires padding and bounding box to symbolic multiples.
  // (void)rewriteAsPaddedOp(rewriter, *tiledLinalgOp, options.tilingOptions);

  return buildTiledLoop(rewriter, std::move(*tiledLinalgOp));
}

LogicalResult mlir::linalg::TileAndDistributePattern::matchAndRewrite(
    Operation *op, PatternRewriter &rewriter) const {
  LinalgOp linalgOp = dyn_cast<LinalgOp>(op);
  if (!linalgOp || !linalgOp.hasTensorSemantics()) return failure();
  if (failed(filter.checkAndNotify(rewriter, linalgOp))) return failure();

  if (linalgOp->getParentOfType<TiledLoopOp>()) return failure();

  Optional<TiledLoopOp> tiledLoopOp =
      tileAndDistributeLinalgOp(rewriter, op, options);
  if (!tiledLoopOp) return failure();
  if (tiledLoopOp->getNumResults() > 0)
    rewriter.replaceOp(op, tiledLoopOp->getResults());
  else
    rewriter.eraseOp(op);
  return success();
}

void mlir::linalg::populateTileAndDistributePattern(
    OwningRewritePatternList &patterns, const TileAndDistributeOptions &opts,
    const LinalgTransformationFilter &filter) {
  patterns.insert<mlir::linalg::TileAndDistributePattern>(
      opts, filter, patterns.getContext());
}
