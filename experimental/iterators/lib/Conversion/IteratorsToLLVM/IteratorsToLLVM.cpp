//===-- IteratorsToLLVM.h - Conversion from Iterators to LLVM ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Conversion/IteratorsToLLVM/IteratorsToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"

#include <memory>
#include <sstream>

using namespace mlir;
using namespace mlir::iterators;
using namespace mlir::LLVM;
using namespace std::string_literals;

namespace {
struct ConvertIteratorsToLLVMPass
    : public ConvertIteratorsToLLVMBase<ConvertIteratorsToLLVMPass> {
  void runOnOperation() override;
};
} // namespace

/// Maps types from the Iterators dialect to corresponding types in LLVM.
class IteratorsTypeConverter : public TypeConverter {
public:
  IteratorsTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(convertTupleType);
  }

private:
  /// Maps a TupleType to a corresponding LLVMStructType
  static Optional<Type> convertTupleType(Type type) {
    if (TupleType tupleType = type.dyn_cast<TupleType>()) {
      return LLVMStructType::getNewIdentified(type.getContext(), "tuple",
                                              tupleType.getTypes());
    }
    return llvm::None;
  }
};

/// Return a symbol reference to the printf function, inserting it into the
/// module if necessary.
static FlatSymbolRefAttr lookupOrInsertPrintf(PatternRewriter &rewriter,
                                              ModuleOp module) {
  if (module.lookupSymbol<LLVMFuncOp>("printf"))
    return SymbolRefAttr::get(rewriter.getContext(), "printf");

  // Create a function declaration for printf, the signature is:
  //   * `i32 (i8*, ...)`
  LLVMPointerType charPointerType = LLVMPointerType::get(rewriter.getI8Type());
  LLVMFunctionType printfFunctionType =
      LLVMFunctionType::get(rewriter.getI32Type(), charPointerType,
                            /*isVarArg=*/true);

  // Insert the printf function into the body of the parent module.
  PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  rewriter.create<LLVMFuncOp>(module.getLoc(), "printf", printfFunctionType);
  return SymbolRefAttr::get(rewriter.getContext(), "printf");
}

/// Return a value representing an access into a global string with the given
/// name, creating the string if necessary.
static Value getOrCreateGlobalString(OpBuilder &builder, Twine name,
                                     Twine value, ModuleOp module) {
  Location loc = module->getLoc();

  // Create the global at the entry of the module.
  GlobalOp global;
  StringAttr nameAttr = builder.getStringAttr(name);
  if (!(global = module.lookupSymbol<GlobalOp>(nameAttr))) {
    StringAttr valueAttr = builder.getStringAttr(value);
    LLVMArrayType globalType =
        LLVMArrayType::get(builder.getI8Type(), valueAttr.size());
    OpBuilder::InsertionGuard insertGuard(builder);
    builder.setInsertionPointToStart(module.getBody());
    global = builder.create<GlobalOp>(loc, globalType, /*isConstant=*/true,
                                      Linkage::Internal, nameAttr, valueAttr,
                                      /*alignment=*/0);
  }

  // Get the pointer to the first character in the global string.
  Value globalPtr = builder.create<AddressOfOp>(loc, global);
  Value zero = builder.create<ConstantOp>(loc, builder.getI64Type(),
                                          builder.getI64IntegerAttr(0));
  return builder.create<GEPOp>(loc, LLVMPointerType::get(builder.getI8Type()),
                               globalPtr, ArrayRef<Value>({zero, zero}));
}

struct ConstantTupleLowering : public ConversionPattern {
  ConstantTupleLowering(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.constant", benefit,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Convert tuple type.
    assert(op->getNumResults() == 1);
    Type structType = typeConverter->convertType(op->getResult(0).getType());

    // Undef.
    Value structValue = rewriter.create<UndefOp>(op->getLoc(), structType);

    // Insert values.
    ArrayAttr values = op->getAttr("values").dyn_cast<ArrayAttr>();
    assert(values);
    for (int i = 0; i < static_cast<int>(values.size()); i++) {
      // Create index attribute.
      ArrayAttr indicesAttr = rewriter.getIndexArrayAttr({i});

      // Create constant value op.
      Value valueOp = rewriter.create<LLVM::ConstantOp>(
          op->getLoc(), values[i].getType(), values[i]);

      // Insert into struct.
      structValue = rewriter.create<InsertValueOp>(op->getLoc(), structValue,
                                                   valueOp, indicesAttr);
    }

    rewriter.replaceOp(op, structValue);

    return success();
  }
};

struct PrintOpLowering : public ConversionPattern {
  PrintOpLowering(TypeConverter &typeConverter, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : ConversionPattern(typeConverter, "iterators.print", benefit, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.size() == 1);
    LLVMStructType structType =
        operands[0].getType().dyn_cast<LLVMStructType>();
    assert(structType);

    // Assemble format string in the form `(%i, %i, ...)`.
    std::string format("(");
    llvm::raw_string_ostream formatStream(format);
    llvm::interleaveComma(structType.getBody(), formatStream, [&](Type type) {
      assert(type == rewriter.getI32Type() && "Only I32 is supported for now");
      formatStream << "%i";
    });
    format += ")\n\0"s;

    // Insert format string as global.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Value formatSpec = getOrCreateGlobalString(
        rewriter, Twine("frmt_spec.") + structType.getName(), format, module);

    // Assemble arguments to printf.
    SmallVector<Value, 8> values = {formatSpec};
    ArrayRef<Type> fieldTypes = structType.getBody();
    for (int i = 0; i < static_cast<int>(fieldTypes.size()); i++) {
      // Create index attribute.
      ArrayAttr indicesAttr = rewriter.getIndexArrayAttr({i});

      // Extract from struct.
      Value value = rewriter.create<ExtractValueOp>(op->getLoc(), fieldTypes[i],
                                                    operands[0], indicesAttr);

      values.push_back(value);
    }

    // Generate call to printf.
    FlatSymbolRefAttr printfRef = lookupOrInsertPrintf(rewriter, module);
    rewriter.create<LLVM::CallOp>(op->getLoc(), rewriter.getI32Type(),
                                  printfRef, values);
    rewriter.eraseOp(op);

    return success();
  }
};

void mlir::iterators::populateIteratorsToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<ConstantTupleLowering, PrintOpLowering>(typeConverter,
                                                       patterns.getContext());
}

void ConvertIteratorsToLLVMPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<func::FuncDialect, LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());
  IteratorsTypeConverter typeConverter;
  populateIteratorsToLLVMConversionPatterns(patterns, typeConverter);
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertIteratorsToLLVMPass() {
  return std::make_unique<ConvertIteratorsToLLVMPass>();
}
