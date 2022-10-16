//===-- Iterators.cpp - Iterators dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Dialect/Tabular/IR/Tabular.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Iterators dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining for iterators
/// dialect operations.
struct IteratorsInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All iterators dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void IteratorsDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc"
      >();
  addInterfaces<IteratorsInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Iterators interfaces
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpInterfaces.cpp.inc"
#include "iterators/Dialect/Iterators/IR/IteratorsTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Iterators operations
//===----------------------------------------------------------------------===//

/// Implement SymbolUserOpInterface.
LogicalResult
AccumulateOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  MLIRContext *context = getContext();
  Type inputElementType =
      getInput().getType().dyn_cast<StreamType>().getElementType();
  Type resultElementType =
      getResult().getType().dyn_cast<StreamType>().getElementType();

  // Verify accumulate function.
  func::FuncOp accumulateFunc = getAccumulateFunc();
  if (!accumulateFunc)
    return emitOpError() << "uses the symbol '" << getAccumulateFuncRef()
                         << "', which does not reference a valid function";

  Type accumulateFuncType = FunctionType::get(
      context, {resultElementType, inputElementType}, resultElementType);
  if (accumulateFunc.getFunctionType() != accumulateFuncType) {
    return emitOpError() << "uses the symbol '" << getAccumulateFuncRef()
                         << "', which does not refer to a function of type "
                         << accumulateFuncType;
  }

  return success();
}

static ParseResult parseAccumulateInitValType(AsmParser & /*parser*/,
                                              Type &initValType,
                                              Type resultType) {
  auto resultStreamType = resultType.cast<StreamType>();
  initValType = resultStreamType.getElementType();
  return success();
}

static void printAccumulateInitValType(AsmPrinter & /*printer*/,
                                       Operation * /*op*/, Type /*initValType*/,
                                       Type /*resultType*/) {}

static ParseResult parseInsertValueType(AsmParser & /*parser*/, Type &valueType,
                                        Type stateType, IntegerAttr indexAttr) {
  int64_t index = indexAttr.getValue().getSExtValue();
  auto castedStateType = stateType.cast<StateType>();
  valueType = castedStateType.getFieldTypes()[index];
  return success();
}

static void printInsertValueType(AsmPrinter & /*printer*/, Operation * /*op*/,
                                 Type /*valueType*/, Type /*stateType*/,
                                 IntegerAttr /*indexAttr*/) {}

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc"

LogicalResult ExtractValueOp::inferReturnTypes(
    MLIRContext * /*context*/, Optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  auto stateType = operands[0].getType().cast<StateType>();
  auto indexAttr = attributes.getAs<IntegerAttr>("index");
  int64_t index = indexAttr.getValue().getSExtValue();
  Type fieldType = stateType.getFieldTypes()[index];
  inferredReturnTypes.assign({fieldType});
  return success();
}

//===----------------------------------------------------------------------===//
// Iterators types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc"
