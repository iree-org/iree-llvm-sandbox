//===-- TritonToLLVM.cpp - Conversion from Triton to LLVM -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TritonToLLVM/TritonToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace mlir::LLVM;
using namespace triton;

namespace {
struct ConvertTritonToLLVMPass
    : public ConvertTritonToLLVMBase<ConvertTritonToLLVMPass> {
  void runOnOperation() override;
};

/// Replaces an op of type SourceOp to an op of type TargetOp while preserving
/// all types, operands, attributes, successors, regions, and its location.
template <typename SourceOp, typename TargetOp>
struct OneToOneOpConversion : public OpConversionPattern<SourceOp> {
  OneToOneOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                       PatternBenefit benefit = 1)
      : OpConversionPattern<SourceOp>(typeConverter, context, benefit) {}

  using OpAdaptor = typename OpConversionPattern<SourceOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
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

void mlir::populateTritonToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<
      // clang-format off
      OneToOneOpConversion<triton::CallOp, func::CallOp>,
      OneToOneOpConversion<triton::FuncOp, func::FuncOp>,
      OneToOneOpConversion<triton::ReturnOp, func::ReturnOp>
      // clang-format on
      >(typeConverter, patterns.getContext());
}

void ConvertTritonToLLVMPass::runOnOperation() {
  auto module = getOperation();
  LLVMTypeConverter typeConverter(&getContext());

  // Convert the remaining ops of this dialect using dialect conversion.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());

  // Lower tt.func op and friends to corresponding ops from func.
  populateTritonToLLVMConversionPatterns(patterns, typeConverter);

  // Lower ops from func to LLVM.
  populateFuncToLLVMFuncOpConversionPattern(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);

  // Use UnrealizedConversionCast as materializations, which have to be cleaned
  // up by later passes.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return Optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertTritonToLLVMPass() {
  return std::make_unique<ConvertTritonToLLVMPass>();
}
