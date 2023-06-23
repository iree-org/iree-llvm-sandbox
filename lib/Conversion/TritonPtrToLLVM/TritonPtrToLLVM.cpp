//===-- TritonPtrToLLVM.cpp - Convert Triton ptr ops to LLVM ----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TritonPtrToLLVM/TritonPtrToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::LLVM;
using namespace triton;

namespace {
struct ConvertTritonPtrToLLVMPass
    : public ConvertTritonPtrToLLVMBase<ConvertTritonPtrToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

void mlir::populateTritonPtrToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  // patterns.add<
  //     // clang-format off
  //     // clang-format on
  //     >(patterns.getContext());
}

void ConvertTritonPtrToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // We have two type converters: the `TypeConverter` instance should only
  // convert pointer types -- including the pointee types if they are pointee
  // types but not if they are scalars outside of a pointer. We thus pass around
  // the `TypeConverter` in the pass and, to implement its conversions, use the
  // `LLVMTypeConverter` for the pointee types.
  TypeConverter typeConverter;
  LLVMTypeConverter llvmTypeConverter(&getContext());

  // Leave unrelated types intact.
  typeConverter.addConversion([&](Type type) { return type; });

  // Scalar `triton::PointerType`.
  auto convertTritonPtrType =
      [&](triton::PointerType type) -> std::optional<Type> {
    Type pointeeType = type.getPointeeType();
    Type convertedPointeeType = llvmTypeConverter.convertType(pointeeType);

    if (!LLVMPointerType::isValidElementType(convertedPointeeType))
      return std::nullopt;

    // TODO(ingomueller): We preserve the address space attribute here but we'll
    //     probably ignore its value in the conversions that use these pointers,
    //     so we'll have to revisit the whole concept of address spaces at some
    //     point.
    return LLVMPointerType::get(convertedPointeeType, type.getAddressSpace());
  };
  typeConverter.addConversion(convertTritonPtrType);
  llvmTypeConverter.addConversion(convertTritonPtrType);

  // Tensor of `triton::PointerType`.
  // TODO(ingomueller): This converts a pointer to an LLVM pointer. While this
  //    covers the general case, very often the pointers belong to a single
  //    allocation, which could be represented as a base pointer and a tensor of
  //    offsets. That, in turn, would preserve the semantics about the loads
  //    being local to each other and maybe fit to (to be developped) primitives
  //    in the indexing dialect.
  typeConverter.addConversion([&](RankedTensorType type) -> Type {
    auto elementType = type.getElementType();
    if (elementType.isa<triton::PointerType>()) {
      auto convertedElementType = llvmTypeConverter.convertType(elementType);
      return RankedTensorType::get(type.getShape(), convertedElementType);
    }
    return type;
  });

  // Convert all triton ops that use `triton::PointerType` operands or results.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVMDialect>();
  target.addDynamicallyLegalDialect<TritonDialect>([&](Operation *op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());

  // Load patterns specific to this pass.
  populateTritonPtrToLLVMConversionPatterns(patterns, typeConverter);

  // Add patterns that converts function signature and calls.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Force application of that pattern if signature is not legal yet.
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return typeConverter.isLegal(op.getOperandTypes());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return typeConverter.isSignatureLegal(op.getCalleeType());
  });

  // Add patterns that convert the types in SCF constructs.
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);

  // Use UnrealizedConversionCast as materializations, which have to be cleaned
  // up by later passes.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return std::optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTritonPtrToLLVMPass() {
  return std::make_unique<ConvertTritonPtrToLLVMPass>();
}
