//===- Transforms.h - LinalgExt transformations as patterns -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace scf {
class ForOp;
}

namespace linalg_ext {

/// Pattern to tile a TilingInterface op using a linalg_ext::TileOp.
struct LinalgExtTilingPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  LinalgExtTilingPattern(MLIRContext *context, linalg::LinalgTilingOptions opt)
      : OpInterfaceRewritePattern<TilingInterface>(context), options(opt) {}

  FailureOr<Operation *>
  returningMatchAndRewrite(TilingInterface op, PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(op, rewriter);
  }

private:
  linalg::LinalgTilingOptions options;
};

/// Pattern to rewrite a linalg_ext::TileOp to an scf::ForOp.
struct TileOpToSCFRewriter : public OpRewritePattern<linalg_ext::TileOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<scf::ForOp>
  returningMatchAndRewrite(linalg_ext::TileOp tileOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(linalg_ext::TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(tileOp, rewriter);
  }
};

/// Pattern to rewrite a linalg_ext::TileOp to a linalg_ext::InParallelOp.
struct TileOpToInParallelRewriter
    : public OpRewritePattern<linalg_ext::TileOp> {
  using OpRewritePattern::OpRewritePattern;

  FailureOr<linalg_ext::InParallelOp>
  returningMatchAndRewrite(linalg_ext::TileOp tileOp,
                           PatternRewriter &rewriter) const;

  LogicalResult matchAndRewrite(linalg_ext::TileOp tileOp,
                                PatternRewriter &rewriter) const override {
    return returningMatchAndRewrite(tileOp, rewriter);
  }
};

} // namespace linalg_ext
} // namespace mlir

#endif // MLIR_DIALECT_LINALGEXT_TRANSFORMS_TRANSFORMS_H
