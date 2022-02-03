//===- TileToInParallel.cpp.cpp - Rewrite TileOp as InParallel -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "Dialects/LinalgExt/PassDetail.h"
#include "Dialects/LinalgExt/Passes.h"
#include "Dialects/LinalgExt/Transforms/Utils.h"
#include "Transforms/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::linalg_ext;

namespace {

struct TileOpToInParallelRewriter
    : public OpRewritePattern<linalg_ext::TileOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg_ext::TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    // TODO: verifier.
    assert(tileOp.getNumResults() > 0 &&
           tileOp.outs().size() == tileOp.getNumResults());

    // TODO: when supported, iterate over the tensor of sizes. This will be
    // iterating through a level of indirection.

    int64_t tiledDim = tileOp.tiled_dim();

    // Construct the loop bounds based on the canonical arithmetic progression.
    Location loc = tileOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value tiledDimValue =
        rewriter.create<arith::ConstantIndexOp>(loc, tiledDim);
    Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value totalSize = rewriter.create<tensor::DimOp>(loc, tileOp.outs().front(),
                                                     tiledDimValue);
    Value step = tileOp.tile_size();
    assert(step.getType().isa<IndexType>() && "NYI: not an index type");

    using AV = AffineValueExpr;
    AffineBuilder ab(rewriter, loc);
    AffineExpr i, j, M;
    bindDims(rewriter.getContext(), i, j);
    bindSymbols(rewriter.getContext(), M);
    Value numThreads = ab.ceil(AV(i).bind(totalSize), AV(M).bind(step));

    // Construct the op without a body builder: we need to clone the ops in the
    // body explicitly after having access to the new bbArgs.
    // As a consequence, `ensureTerminator` is not called and the body has no
    // terminator.
    linalg_ext::InParallelOp inParallelOp =
        rewriter.create<linalg_ext::InParallelOp>(loc, tileOp->getResultTypes(),
                                                  numThreads);

    // At the beginning of the InParallelOp, compute offset and sizes.
    rewriter.setInsertionPointToStart(inParallelOp.getBody());

    // Materialize the implicit subtensors as explicit subset_extract.
    // TODO: generalize to multiple offset/chunk_size bbargs if needed.
    // TODO: generalize the subset op.
    SmallVector<Value> leadingOffsets, leadingSizes, leadingStrides;
    for (int64_t i = 0; i < tiledDim; ++i) {
      leadingOffsets.push_back(zero);
      leadingSizes.push_back(
          rewriter.createOrFold<tensor::DimOp>(loc, tileOp.outs().front(), i));
      leadingStrides.push_back(one);
    }
    // clang-format off
    Value offset = ab.mul(AV(i).bind(inParallelOp.getThreadIndex()), 
                          AV(M).bind(step));
    Value size = ab.min(
      ValueRange{ab.sub(AV(i).bind(totalSize), AV(j).bind(offset)),
      step});
    // clang-format on
    leadingOffsets.push_back(offset);
    leadingSizes.push_back(size);
    leadingStrides.push_back(one);

    SmallVector<Value> implicitSubtensorExtracts;
    for (Value tensor : tileOp.outs()) {
      implicitSubtensorExtracts.push_back(
          createSubsetExtractOpFromLeadingOffsetsSizesAndStrides(
              rewriter, loc, tensor, leadingOffsets, leadingSizes,
              leadingStrides));
    }

    // Get a reference to the TileOp terminator before the body is merged and it
    // becomes too hard to get to the terminator.
    auto tileYieldOp = cast<TileYieldOp>(tileOp.getBody()->getTerminator());

    // Regroup the values that replace the tileOp's bbArg and move the body.
    SmallVector<Value> bbArgsTranslated{offset, size};
    llvm::append_range(bbArgsTranslated, implicitSubtensorExtracts);
    rewriter.mergeBlockBefore(&tileOp.region().front(),
                              inParallelOp.getBody()->getTerminator(),
                              bbArgsTranslated);

    // tileOp's terminator is not the terminator, insert explicit subset_insert
    // ops and feed them to a new scf.yield terminator that we can now add.
    PerformConcurrentlyOp performConcurrentlyOp = inParallelOp.getTerminator();

    for (auto it : llvm::zip(tileYieldOp->getOperands(), tileOp.outs())) {
      SmallVector<Value> offsets, sizes, strides;
      completeOffsetsSizesAndStrides(rewriter, loc, std::get<0>(it),
                                     leadingOffsets, leadingSizes,
                                     leadingStrides, offsets, sizes, strides);
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(
          performConcurrentlyOp.getBody()->getTerminator());
      createParallelInsertSliceOpFromLeadingOffsetsSizesAndStrides(
          rewriter, loc, std::get<0>(it), std::get<1>(it), offsets, sizes,
          strides);
    }

    // Cleanup and replace.
    rewriter.eraseOp(tileYieldOp);
    rewriter.replaceOp(tileOp, inParallelOp.getResults());

    return success();
  }
};

struct TileToInParallelPass
    : public TileToInParallelBase<TileToInParallelPass> {
  void runOnOperation() override;
};
} // namespace

void TileToInParallelPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  MLIRContext *context = funcOp.getContext();
  RewritePatternSet patterns(context);
  patterns.insert<TileOpToInParallelRewriter>(context);
  (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::linalg_ext::createTileToInParallelPass() {
  return std::make_unique<TileToInParallelPass>();
}
