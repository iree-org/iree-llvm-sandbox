//===-- DataFlowToIterators.h - DataFlow to Iterators conversion-*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Conversion/DataFlowToIterators/DataFlowToIterators.h"

#include "../PassDetail.h"
#include "iterators/Dialect/DataFlow/IR/DataFlow.h"
#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

#include <memory>

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::func;
using namespace mlir::LLVM;

namespace {
struct ConvertDataFlowToIteratorsPass
    : public ConvertDataFlowToIteratorsBase<ConvertDataFlowToIteratorsPass> {
  void runOnOperation() override;
};
} // namespace

/// Maps StreamType to llvm.ptr<i8>.
class DataFlowTypeConverter : public TypeConverter {
public:
  DataFlowTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertStreamType);
  }

private:
  /// Maps StreamType to llvm.ptr<i8>.
  static Optional<Type> convertStreamType(Type type) {
    if (type.isa<dataflow::StreamType>())
      return LLVM::LLVMPointerType::get(IntegerType::get(type.getContext(), 8));
    return llvm::None;
  }
};

/// Returns or creates a function declaration at the module of the provided
/// original op.
FuncOp lookupOrCreateFuncOp(llvm::StringRef fnName, FunctionType fnType,
                            Operation *op, PatternRewriter &rewriter) {
  ModuleOp module = op->getParentOfType<ModuleOp>();
  assert(module);

  // Return function if already declared.
  if (FuncOp funcOp = module.lookupSymbol<FuncOp>(fnName))
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
struct CppIteratorConversionPattern : public ConversionPattern {
  CppIteratorConversionPattern(TypeConverter &typeConverter,
                               MLIRContext *context, StringRef rootName,
                               StringRef constructorName,
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

void mlir::dataflow::populateDataFlowToIteratorsConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<CppIteratorConversionPattern>(
      typeConverter, patterns.getContext(), "dataflow.sampleInput",
      "iteratorsMakeSampleInputOperator",
      "iteratorsDestroySampleInputOperator");
  patterns.add<CppIteratorConversionPattern>(
      typeConverter, patterns.getContext(), "dataflow.reduce",
      "iteratorsMakeReduceOperator", "iteratorsDestroyReduceOperator");
  patterns.add<CppIteratorConversionPattern>(
      typeConverter, patterns.getContext(), "dataflow.sink",
      "iteratorsComsumeAndPrint", "_dummy");
}

void ConvertDataFlowToIteratorsPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());
  DataFlowTypeConverter typeConverter;
  populateDataFlowToIteratorsConversionPatterns(patterns, typeConverter);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertDataFlowToIteratorsPass() {
  return std::make_unique<ConvertDataFlowToIteratorsPass>();
}
