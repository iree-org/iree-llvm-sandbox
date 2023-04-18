//===-- StatesToLLVM.cpp - Convert Iterator states to LLVM ------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/StatesToLLVM/StatesToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "structured/Dialect/Iterators/IR/Iterators.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace mlir::structured;
using namespace mlir::LLVM;

namespace {
struct ConvertStatesToLLVMPass
    : public ConvertStatesToLLVMBase<ConvertStatesToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

/// Maps state types from the Iterators dialect to corresponding types in LLVM.
class StateTypeConverter : public TypeConverter {
public:
  StateTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([&](Type type) { return convertIteratorStateType(type); });
  }

private:
  /// Maps an iterator state type to a corresponding LLVMStructType.
  Optional<Type> convertIteratorStateType(Type type) {
    if (auto stateType = type.dyn_cast<StateType>()) {
      llvm::SmallVector<Type> fieldTypes(stateType.getFieldTypes().begin(),
                                         stateType.getFieldTypes().end());
      for (auto &type : fieldTypes) {
        type = convertType(type);
      }
      return LLVMStructType::getLiteral(type.getContext(), fieldTypes);
    }
    return std::nullopt;
  }
};

struct CreateStateOpLowering : public OpConversionPattern<CreateStateOp> {
  CreateStateOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(CreateStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Create undef struct.
    Type stateType = op.getResult().getType();
    Type structType = getTypeConverter()->convertType(stateType);
    Value state = rewriter.create<LLVM::UndefOp>(loc, structType);

    // Insert values.
    for (auto [index, value] : llvm::enumerate(adaptor.getValues())) {
      state = rewriter.create<LLVM::InsertValueOp>(loc, state, value, index);
    }

    // `state` now constists of a struct with all fields set.
    rewriter.replaceOp(op, state);

    return success();
  }
};

struct ExtractValueOpLowering
    : public OpConversionPattern<structured::ExtractValueOp> {
  ExtractValueOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                         PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(structured::ExtractValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type resultType = getTypeConverter()->convertType(op.getResult().getType());
    Value value = rewriter.create<LLVM::ExtractValueOp>(
        loc, resultType, adaptor.getState(), adaptor.getIndex().getSExtValue());
    rewriter.replaceOp(op, value);
    return success();
  }
};

struct InsertValueOpLowering
    : public OpConversionPattern<structured::InsertValueOp> {
  InsertValueOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(structured::InsertValueOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Value updatedState = rewriter.create<LLVM::InsertValueOp>(
        loc, adaptor.getState(), adaptor.getValue(),
        adaptor.getIndex().getSExtValue());
    rewriter.replaceOp(op, updatedState);
    return success();
  }
};

void mlir::structured::populateStatesToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<
      // clang-format off
      CreateStateOpLowering,
      ExtractValueOpLowering,
      InsertValueOpLowering
      // clang-format on
      >(typeConverter, patterns.getContext());
}

void ConvertStatesToLLVMPass::runOnOperation() {
  auto module = getOperation();
  StateTypeConverter typeConverter;

  // Convert the remaining ops of this dialect using dialect conversion.
  ConversionTarget target(getContext());
  target.addLegalDialect<LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());

  populateStatesToLLVMConversionPatterns(patterns, typeConverter);

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
    return Optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertStatesToLLVMPass() {
  return std::make_unique<ConvertStatesToLLVMPass>();
}
