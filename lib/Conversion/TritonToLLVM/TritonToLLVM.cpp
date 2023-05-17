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
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
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

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  LoadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle unmasked pointers for now.
    if (op.getMask() || op.getOther())
      return failure();

    Type ptrType = op.getPtr().getType();

    // Only handle scalar pointers to numerics for now.
    if (auto ttPtrType = ptrType.dyn_cast<triton::PointerType>()) {
      if (ttPtrType.getPointeeType().isIntOrIndexOrFloat()) {
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, adaptor.getPtr());
        return success();
      }
    }

    return failure();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  StoreOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle unmasked pointers for now.
    if (op.getMask())
      return failure();

    Type ptrType = op.getPtr().getType();

    // Only handle scalar pointers to numerics for now.
    if (auto ttPtrType = ptrType.dyn_cast<triton::PointerType>()) {
      if (ttPtrType.getPointeeType().isIntOrIndexOrFloat()) {
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                                   adaptor.getPtr());
        return success();
      }
    }

    return failure();
  }
};
} // namespace

void mlir::populateTritonToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<
      // clang-format off
      LoadOpConversion,
      StoreOpConversion,
      OneToOneOpConversion<triton::CallOp, func::CallOp>,
      OneToOneOpConversion<triton::FuncOp, func::FuncOp>,
      OneToOneOpConversion<triton::ReturnOp, func::ReturnOp>
      // clang-format on
      >(typeConverter, patterns.getContext());
}

void ConvertTritonToLLVMPass::runOnOperation() {
  auto module = getOperation();
  LLVMTypeConverter typeConverter(&getContext());

  // triton::PointerType: Replicate logic from
  // TritonGPUToLLVMTypeConverter::convertTritonPointerType.
  // TODO(ingomueller): We preserve the address space attribute here but we'll
  //     probably ignore its value in the conversions that use these pointers,
  //     so we'll have to revisit the whole concept of address spaces at some
  //     point.
  typeConverter.addConversion([&](triton::PointerType type) {
    return LLVM::LLVMPointerType::get(
        typeConverter.convertType(type.getPointeeType()),
        type.getAddressSpace());
  });

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

  // Add patterns that converts function signature and calls.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

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