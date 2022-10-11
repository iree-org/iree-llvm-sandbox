//===-- TabularToLLVM.h - Conversion from Tabular to LLVM -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Conversion/TabularToLLVM/TabularToLLVM.h"

#include "../PassDetail.h"
#include "iterators/Dialect/Tabular/IR/Tabular.h"
#include "iterators/Utils/MLIRSupport.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace mlir::iterators;
using namespace mlir::LLVM;

namespace {
struct ConvertTabularToLLVMPass
    : public ConvertTabularToLLVMBase<ConvertTabularToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

/// Maps types from the Tabular dialect to corresponding types in LLVM.
class TabularTypeConverter : public TypeConverter {
public:
  TabularTypeConverter(LLVMTypeConverter &llvmTypeConverter)
      : llvmTypeConverter(llvmTypeConverter) {
    addConversion([](Type type) { return type; });
    addConversion(convertTabularViewType);

    // Convert MemRefType using LLVMTypeConverter.
    addConversion([&](Type type) -> llvm::Optional<Type> {
      if (type.isa<MemRefType>())
        return llvmTypeConverter.convertType(type);
      return llvm::None;
    });
  }

private:
  /// Maps a TabularViewType to an LLVMStruct of pointers, i.e., to a "struct of
  /// arrays".
  static Optional<Type> convertTabularViewType(Type type) {
    if (auto viewType = type.dyn_cast<TabularViewType>()) {
      Type i64 = IntegerType::get(type.getContext(), /*width=*/64);
      SmallVector<Type> fieldTypes{i64};
      fieldTypes.reserve(viewType.getNumColumnTypes() + 1);
      llvm::transform(viewType.getColumnTypes(), std::back_inserter(fieldTypes),
                      [](Type t) { return LLVMPointerType::get(t); });
      return LLVMStructType::getLiteral(type.getContext(), fieldTypes);
    }
    return llvm::None;
  }

  LLVMTypeConverter llvmTypeConverter;
};

void mlir::iterators::populateTabularToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {}

void ConvertTabularToLLVMPass::runOnOperation() {
  auto module = getOperation();
  LLVMTypeConverter llvmTypeConverter(&getContext());
  TabularTypeConverter typeConverter(llvmTypeConverter);

  // Convert the remaining ops of this dialect using dialect conversion.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());

  populateTabularToLLVMConversionPatterns(patterns, typeConverter);

  // Add patterns that converts function signature and calls.
  populateFunctionOpInterfaceTypeConversionPattern<FuncOp>(patterns,
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

  // Use UnrealizedConversionCast as materializations, which have to be cleaned
  // up by later passes.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return Optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTabularToLLVMPass() {
  return std::make_unique<ConvertTabularToLLVMPass>();
}
