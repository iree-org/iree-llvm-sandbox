//===-- TritonFuncToFunc.cpp - Convert Triton func ops to func --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TritonFuncToFunc/TritonFuncToFunc.h"

#include "../PassDetail.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace triton;

namespace {
struct ConvertTritonFuncToFuncPass
    : public ConvertTritonFuncToFuncBase<ConvertTritonFuncToFuncPass> {
  void runOnOperation() override;
};

/// Replaces an op of type SourceOp to an op of type TargetOp while preserving
/// all types, operands, attributes, successors, regions, and its location.
template <typename SourceOp, typename TargetOp>
struct OneToOneOpReplacementPattern : public OpRewritePattern<SourceOp> {
  OneToOneOpReplacementPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern<SourceOp>(context, benefit) {}

  LogicalResult matchAndRewrite(SourceOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = this->getContext();
    SmallVector<std::unique_ptr<Region>> regions;
    for (auto &region : op->getRegions()) {
      auto &newRegion = regions.emplace_back(new Region);
      rewriter.inlineRegionBefore(region, *newRegion, newRegion->end());
    }
    Operation *newOp = rewriter.create(
        op->getLoc(), StringAttr::get(context, TargetOp::getOperationName()),
        op->getOperands(), op->getResultTypes(), op->getAttrs(),
        op->getSuccessors(), regions);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};
} // namespace

void mlir::populateTritonFuncToFuncConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      OneToOneOpReplacementPattern<triton::CallOp, func::CallOp>,
      OneToOneOpReplacementPattern<triton::FuncOp, func::FuncOp>,
      OneToOneOpReplacementPattern<triton::ReturnOp, func::ReturnOp>
      // clang-format on
      >(patterns.getContext());
}

void ConvertTritonFuncToFuncPass::runOnOperation() {
  auto module = getOperation();

  // Lower tt.func op and friends to corresponding ops from func.
  RewritePatternSet patterns(&getContext());
  populateTritonFuncToFuncConversionPatterns(patterns);

  // Mark the three func ops in the Triton dialect as illegal; everything else
  // is legal.
  ConversionTarget target(getContext());
  target.addLegalDialect<FuncDialect, TritonDialect>();
  target.addIllegalOp<triton::CallOp, triton::FuncOp, triton::ReturnOp>();
  target.addLegalOp<ModuleOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTritonFuncToFuncPass() {
  return std::make_unique<ConvertTritonFuncToFuncPass>();
}
