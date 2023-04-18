//===-- Tuple.cpp - Tuple dialect -------------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Tuple/IR/Tuple.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace mlir;
using namespace mlir::tuple;

//===----------------------------------------------------------------------===//
// Tuple dialect.
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Tuple/IR/TupleOpsDialect.cpp.inc"

namespace {
/// This class defines the interface for handling inlining for tuple dialect
/// operations.
struct TupleInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  /// All Tuple dialect ops can be inlined.
  bool isLegalToInline(Operation *, Region *, bool, IRMapping &) const final {
    return true;
  }
};
} // namespace

void TupleDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "structured/Dialect/Tuple/IR/TupleOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "structured/Dialect/Tuple/IR/TupleOpsTypes.cpp.inc"
      >();
  addInterfaces<TupleInlinerInterface>();
}

//===----------------------------------------------------------------------===//
// Tuple operations.
//===----------------------------------------------------------------------===//

static ParseResult parseTupleElementTypes(AsmParser &parser,
                                          SmallVectorImpl<Type> &elementsTypes,
                                          Type type) {
  assert(type.isa<TupleType>());
  auto tupleType = type.cast<TupleType>();
  elementsTypes.append(tupleType.begin(), tupleType.end());
  return success();
}

static void printTupleElementTypes(AsmPrinter &printer, Operation *op,
                                   TypeRange elementsTypes, Type tupleType) {}

#define GET_OP_CLASSES
#include "structured/Dialect/Tuple/IR/TupleOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Tuple types.
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "structured/Dialect/Tuple/IR/TupleOpsTypes.cpp.inc"
