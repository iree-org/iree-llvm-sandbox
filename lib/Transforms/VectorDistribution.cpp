//===- VectorDistribution.cpp - Vector distribution experiements--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "Passes.h"
#include "Transforms/Transforms.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct PropagateVectorDistributionPass
    : public VectorPropagateDistributionBase<PropagateVectorDistributionPass> {
  PropagateVectorDistributionPass() = default;
  PropagateVectorDistributionPass(const PropagateVectorDistributionPass &pass) {
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  /// Function pass entry point.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    vector::populatePropagateVectorDistributionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
mlir::createPropagateVectorDistribution() {
  return std::make_unique<PropagateVectorDistributionPass>();
}
