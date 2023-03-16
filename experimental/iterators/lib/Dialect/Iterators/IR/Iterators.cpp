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

static ParseResult
parsePrintOpArgs(OpAsmParser &parser, StringAttr &prefixAttr,
                 Optional<OpAsmParser::UnresolvedOperand> &elementOperand,
                 StringAttr &suffixAttr, Type &elementType) {
  if (failed(parser.parseLParen()))
    return failure();

  bool expectOtherArgument = false;

  // Try to parse prefix attribute.
  OptionalParseResult prefixParseResult =
      parser.parseOptionalAttribute(prefixAttr);
  if (prefixParseResult.has_value()) {
    if (failed(prefixParseResult.value()))
      return failure();

    // If there is a prefix and a comma, we expect to parse another argument.
    if (succeeded(parser.parseOptionalComma()))
      expectOtherArgument = true;
  }

  // Try to parse an operand.
  OpAsmParser::UnresolvedOperand element;
  OptionalParseResult operandParseResult = parser.parseOptionalOperand(element);
  if (operandParseResult.has_value()) {
    if (failed(operandParseResult.value()))
      return failure();
    elementOperand = element;
    expectOtherArgument = false;

    // If there is an operand and a comma, we expect another argument.
    if (succeeded(parser.parseOptionalComma()))
      expectOtherArgument = true;
  }

  // Try to parse a suffix.
  OptionalParseResult suffixParseResult =
      parser.parseOptionalAttribute(suffixAttr);
  if (suffixParseResult.has_value()) {
    if (failed(suffixParseResult.value()))
      return failure();
  } else if (expectOtherArgument) {
    // We expected another argument but didn't find one.
    return parser.emitError(parser.getNameLoc()) << "expected another argument";
    return failure();
  }

  if (failed(parser.parseRParen()))
    return failure();

  // Parse operand type if we had an operand.
  if (elementOperand.has_value()) {
    if (failed(parser.parseColonType(elementType)))
      return failure();
  }

  return success();
}

static void printPrintOpArgs(OpAsmPrinter &printer, Operation *op,
                             StringAttr prefix, Value element,
                             StringAttr suffix, Type type) {
  printer.getStream() << "(";

  // Detect if the two string attributes are ambigous. This may happen if the
  // prefix has the default value and is therefor omitted but the suffix is
  // printed, and there is no element in the middle to disambiguate. In this
  // case, the output would be parsed as *prefix* (and the suffix would be
  // default-valued).
  bool printSuffix = suffix && suffix != "\n";
  bool ambiguousStrings = printSuffix && !element;

  // Print prefix.
  bool needComma = false;
  if (prefix && (!prefix.getValue().empty() || ambiguousStrings)) {
    printer.printAttributeWithoutType(prefix);
    needComma = true;
  }

  // Print element.
  bool needType = false;
  if (element) {
    if (needComma)
      printer.getStream() << ", ";
    printer.printOperand(element);
    needComma = true;
    needType = true;
  }

  // Print suffix.
  if (printSuffix) {
    if (needComma)
      printer.getStream() << ", ";
    printer.printAttributeWithoutType(suffix);
  }

  printer.getStream() << ")";

  // Print type if we printed an element.
  if (needType && type) {
    printer.getStream() << " : ";
    printer.printType(type);
  }
}

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
