//===-- IteratorsToStandard.h - Conv. from Iterators to std -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Conversion/IteratorsToStandard/IteratorsToStandard.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/Support/Casting.h"

#include <memory>

using namespace mlir;
using namespace mlir::iterators;

namespace {
struct ConvertIteratorsToStandardPass
    : public ConvertIteratorsToStandardBase<ConvertIteratorsToStandardPass> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

/// Maps IteratorType to llvm.ptr<i8>.
class IteratorsTypeConverter : public TypeConverter {
public:
  IteratorsTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertIteratorType);
    addConversion(convertOptional);
  }

private:
  /// Maps IteratorType to llvm.ptr<i8>.
  static Optional<Type> convertIteratorType(Type type) {
    if (type.isa<iterators::IteratorType>())
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
    return llvm::None;
  }

  static Optional<Type> convertOptional(Type type) {
    OptionalType optionalType = type.dyn_cast<OptionalType>();
    if (!optionalType)
      return llvm::None;

    Type boolType = IntegerType::get(type.getContext(), 1);
    Type elementType = optionalType.getElementType();
    SmallVector<Type, 4> types = {boolType, elementType};

    auto structType = LLVM::LLVMStructType::getNewIdentified(type.getContext(),
                                                             "Optional", types);

    return structType;
  }
};

//===----------------------------------------------------------------------===//
// Iterators
//===----------------------------------------------------------------------===//

/// Returns or creates a function declaration at the module of the provided
/// original op.
FuncOp lookupOrCreateFuncOp(llvm::StringRef fnName, FunctionType fnType,
                            Operation *op, PatternRewriter &rewriter) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  assert(module);

  // Return function if already declared.
  if (FuncOp funcOp = module.lookupSymbol<mlir::FuncOp>(fnName))
    return funcOp;

  // Add new declaration at the start of the module.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  FuncOp funcOp = rewriter.create<FuncOp>(op->getLoc(), fnName, fnType);
  funcOp.setPrivate();
  return funcOp;
}

/// Replaces an instance of a certain IteratorOp with a call to the given
/// external constructor as well as a call to the given destructor at the end of
/// the block.
struct IteratorConversionPattern : public ConversionPattern {
  IteratorConversionPattern(TypeConverter &typeConverter, MLIRContext *context,
                            StringRef rootName, StringRef constructorName,
                            StringRef destructorName,
                            PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, rootName, benefit, context),
        constructorName(constructorName), destructorName(destructorName) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert result types.
    llvm::SmallVector<Type, 4> resultTypes;
    if (typeConverter->convertTypes(op->getResultTypes(), resultTypes).failed())
      return failure();
    assert(resultTypes.size() <= 1 &&
           "Iterators may have only one output (and sinks have none).");

    // Constructor (aka "iteratorsMake*Operator")

    // Look up or declare function symbol.
    auto const fnType =
        FunctionType::get(getContext(), TypeRange(operands), resultTypes);
    FuncOp funcOp = lookupOrCreateFuncOp(constructorName, fnType, op, rewriter);

    // Replace op with call to function.
    func::CallOp callOp =
        rewriter.replaceOpWithNewOp<func::CallOp>(op, funcOp, operands);

    // Destructor (aka "iteratorsDestroy*Operator")

    // No destructor necessary for sinks.
    if (resultTypes.empty())
      return success();
    assert(resultTypes.size() == 1);

    {
      Value result = callOp.getResult(0);
      assert(result.use_empty() &&
             "Values of type Iterator cannot outlive their consumers, so "
             "functions are not allowed to return them.");

      // Look up or declare function symbol.
      auto const fnType =
          FunctionType::get(getContext(), TypeRange(resultTypes), TypeRange());
      FuncOp funcOp =
          lookupOrCreateFuncOp(destructorName, fnType, op, rewriter);

      // Add call to destructor to the end of the block.
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(callOp->getBlock()->getTerminator());
      rewriter.create<func::CallOp>(op->getLoc(), funcOp, result);
    }

    return success();
  }

private:
  StringRef constructorName;
  StringRef destructorName;
};

//===----------------------------------------------------------------------===//
// Optional
//===----------------------------------------------------------------------===//

/// Replaces an instance of EmptyOptionalOp with the construction of an LLVM
/// struct whose first bit is set to false.
struct EmptyOptionalConversionPattern : public ConversionPattern {
  EmptyOptionalConversionPattern(TypeConverter &typeConverter,
                                 MLIRContext *context,
                                 PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.emptyoptional", benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Create undefined LLVM struct.
    llvm::SmallVector<Type, 4> types;
    LogicalResult res =
        typeConverter->convertTypes(op->getResultTypes(), types);
    assert(res.succeeded());
    auto undefOp = rewriter.create<LLVM::UndefOp>(op->getLoc(), types);

    // Set first field of struct (the valid bit) to false.
    Type boolType = IntegerType::get(getContext(), 1);
    arith::ConstantOp falseOp = rewriter.create<arith::ConstantOp>(
        op->getLoc(), IntegerAttr::get(boolType, 0));

    auto zero = ArrayAttr::get(
        getContext(), {IntegerAttr::get(IndexType::get(getContext()), 0)});
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, undefOp, falseOp,
                                                     zero);
    return success();
  }
};

/// Replaces an instance of InsertValueOp with the corresponding op for LLVM
/// structs.
struct InsertValueConversionPattern : public ConversionPattern {
  InsertValueConversionPattern(TypeConverter &typeConverter,
                               MLIRContext *context, PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.insertvalue", benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 2);
    auto one = ArrayAttr::get(
        getContext(), {IntegerAttr::get(IndexType::get(getContext()), 1)});
    rewriter.replaceOpWithNewOp<LLVM::InsertValueOp>(op, operands[0],
                                                     operands[1], one);
    return success();
  }
};

/// Replaces an instance of ExtractValueOp with the corresponding op for LLVM
/// structs.
struct ExtractValueConversionPattern : public ConversionPattern {
  ExtractValueConversionPattern(TypeConverter &typeConverter,
                                MLIRContext *context,
                                PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.extractvalue", benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    auto one = ArrayAttr::get(
        getContext(), {IntegerAttr::get(IndexType::get(getContext()), 1)});
    assert(op->getNumResults() == 1);
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(
        op, op->getResult(0).getType(), operands[0], one);
    return success();
  }
};

/// Replaces an instance of HasValueOp with the corresponding op for LLVM
/// structs.
struct HasValueConversionPattern : public ConversionPattern {
  HasValueConversionPattern(TypeConverter &typeConverter, MLIRContext *context,
                            PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.hasvalue", benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    auto zero = ArrayAttr::get(
        getContext(), {IntegerAttr::get(IndexType::get(getContext()), 0)});
    Type boolType = IntegerType::get(getContext(), 1);
    rewriter.replaceOpWithNewOp<LLVM::ExtractValueOp>(op, boolType, operands[0],
                                                      zero);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Set-up
//===----------------------------------------------------------------------===//

void mlir::iterators::populateIteratorsToStandardConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<IteratorConversionPattern>(
      typeConverter, patterns.getContext(), "iterators.sampleInput",
      "iteratorsMakeSampleInputOperator",
      "iteratorsDestroySampleInputOperator");
  patterns.add<IteratorConversionPattern>(
      typeConverter, patterns.getContext(), "iterators.reduce",
      "iteratorsMakeReduceOperator", "iteratorsDestroyReduceOperator");
  patterns.add<IteratorConversionPattern>(typeConverter, patterns.getContext(),
                                          "iterators.sink",
                                          "iteratorsComsumeAndPrint", "_dummy");
  patterns.add<EmptyOptionalConversionPattern, InsertValueConversionPattern,
               ExtractValueConversionPattern, HasValueConversionPattern>(
      typeConverter, patterns.getContext());
}

void ConvertIteratorsToStandardPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, memref::MemRefDialect,
                         LLVM::LLVMDialect, arith::ArithmeticDialect>();
  target.addLegalOp<ModuleOp, FuncOp, func::ReturnOp>();
  RewritePatternSet patterns(&getContext());
  IteratorsTypeConverter typeConverter;
  populateIteratorsToStandardConversionPatterns(patterns, typeConverter);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertIteratorsToStandardPass() {
  return std::make_unique<ConvertIteratorsToStandardPass>();
}
