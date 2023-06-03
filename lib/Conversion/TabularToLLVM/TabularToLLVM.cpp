//===-- TabularToLLVM.cpp - Conversion from Tabular to LLVM -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TabularToLLVM/TabularToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "structured/Dialect/Tabular/IR/Tabular.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace mlir::LLVM;
using namespace mlir::tabular;

namespace {
struct ConvertTabularToLLVMPass
    : public ConvertTabularToLLVMBase<ConvertTabularToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

TabularTypeConverter::TabularTypeConverter(LLVMTypeConverter &llvmTypeConverter)
    : llvmTypeConverter(llvmTypeConverter) {
  addConversion([](Type type) { return type; });
  addConversion(convertTabularViewType);

  // Convert MemRefType using LLVMTypeConverter.
  addConversion([&](Type type) -> std::optional<Type> {
    if (type.isa<MemRefType>())
      return llvmTypeConverter.convertType(type);
    return std::nullopt;
  });
}

std::optional<Type> TabularTypeConverter::convertTabularViewType(Type type) {
  if (auto viewType = type.dyn_cast<TabularViewType>()) {
    MLIRContext *context = type.getContext();
    Type dynamicSize = IntegerType::get(context, /*width=*/64);
    SmallVector<Type> fieldTypes{dynamicSize};
    fieldTypes.reserve(viewType.getNumColumnTypes() + 1);
    llvm::transform(viewType.getColumnTypes(), std::back_inserter(fieldTypes),
                    [&](Type t) { return LLVMPointerType::get(context); });
    return LLVMStructType::getLiteral(context, fieldTypes);
  }
  return std::nullopt;
}

/// Lowers view_as_tabular to LLVM IR that extracts the bare pointers and the
/// number of elements from the given memrefs.
///
/// Possible result:
///
/// %1 = builtin.unrealized_conversion_cast %0 :
///        memref<3xi32> to !llvm.struct<(ptr, ptr, i64,
///                                       array<1 x i64>, array<1 x i64>)>
/// %2 = llvm.mlir.undef : !llvm.struct<(i64, ptr)>
/// %3 = llvm.extractvalue %1[1] :
///        !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
/// %4 = llvm.extractvalue %1[3, 0] :
///        !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
/// %5 = llvm.insertvalue %3, %2[1] : !llvm.struct<(i64, ptr)>
/// %6 = llvm.insertvalue %4, %5[0] : !llvm.struct<(i64, ptr)>
struct ViewAsTabularOpLowering : public OpConversionPattern<ViewAsTabularOp> {
  ViewAsTabularOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                          PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(ViewAsTabularOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Create empty struct for view.
    Type viewStructType = typeConverter->convertType(op.getView().getType());
    Value viewStruct = rewriter.create<UndefOp>(loc, viewStructType);

    // Extract column pointers and number of elements.
    Value numElements;
    for (auto [index, operand] : llvm::enumerate(adaptor.getOperands())) {
      assert(op->getOperandTypes()[index]
                 .cast<MemRefType>()
                 .getLayout()
                 .isIdentity());

      // Extract pointer and number of elements from memref descriptor.
      MemRefDescriptor descriptor(operand);
      Value ptr = descriptor.alignedPtr(rewriter, loc);
      if (index == 0) {
        numElements = descriptor.size(rewriter, loc, 0);
      }

      // Insert pointer into view struct.
      viewStruct =
          rewriter.create<LLVM::InsertValueOp>(loc, viewStruct, ptr, index + 1);
    }

    // Insert number of elements.
    viewStruct =
        rewriter.create<LLVM::InsertValueOp>(loc, viewStruct, numElements, 0);

    // Replace original op.
    rewriter.replaceOp(op, {viewStruct});

    return success();
  }
};

void mlir::tabular::populateTabularToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ViewAsTabularOpLowering>(typeConverter, patterns.getContext());
}

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
mlir::createConvertTabularToLLVMPass() {
  return std::make_unique<ConvertTabularToLLVMPass>();
}
