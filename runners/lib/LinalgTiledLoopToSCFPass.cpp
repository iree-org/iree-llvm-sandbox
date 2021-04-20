//===- LinalgTiledLoopToSCFPass.cpp - Tile and distribute to linalg.tile --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements conversion of `linalg.tiled_loop` to `scf.parallel` or
// `scf.for`.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg;

struct TiledLoopConverter : public OpRewritePattern<TiledLoopOp> {
  using OpRewritePattern<TiledLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TiledLoopOp tiledLoop,
                                PatternRewriter& rewriter) const override {
    Location loc = tiledLoop.getLoc();

    // Fail conversion if the `tiled_loop` has not been bufferized.
    bool onBuffers = llvm::all_of(tiledLoop.outputs(), [&](Value arg) {
      return arg.getType().isa<MemRefType>();
    });
    if (!onBuffers) return failure();

    auto moveTiledLoopBody = [&](OpBuilder& builder, Location loc,
                                 ValueRange ivs) {
      BlockAndValueMapping bvm;
      bvm.map(tiledLoop.getInductionVars(), ivs);
      for (auto& op : tiledLoop.getBody()->without_terminator())
        builder.clone(op, bvm);
    };

    scf::buildLoopNest(rewriter, loc, tiledLoop.lowerBound(),
                       tiledLoop.upperBound(), tiledLoop.step(),
                       moveTiledLoopBody);
    rewriter.eraseOp(tiledLoop);
    return success();
  }
};

namespace mlir {
namespace linalg {

void populateTiledLoopsToSCF(OwningRewritePatternList& patterns) {
  patterns.add<TiledLoopConverter>(patterns.getContext());
}

}  // namespace linalg
}  // namespace mlir
