//===-- IteratorsDialect.cpp - Iterators dialect ----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/Iterators.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Iterators dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.cpp.inc"

void IteratorsDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Iterators operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Iterators types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Optional
//===----------------------------------------------------------------------===//

LogicalResult OptionalInsertValueOp::verify() {
  // Check result type.
  if (optional().getType() != result().getType()) {
    return emitOpError() << "Type mismatch: Inserting into "
                         << optional().getType() << " should produce a "
                         << optional().getType() << " but this op returns "
                         << result().getType();
  }

  // Check value type.
  OptionalType inputType = optional().getType().dyn_cast<OptionalType>();
  if (!inputType)
    return failure();

  if (inputType.getElementType() != value().getType()) {
    return emitOpError() << "Type mismatch: Inserting into " << inputType
                         << " requires to insert a "
                         << inputType.getElementType()
                         << " but this op inserts a " << value().getType();
  }

  return success();
}

LogicalResult OptionalExtractValueOp::verify() {
  // Check return type.
  OptionalType inputType = input().getType().dyn_cast<OptionalType>();
  if (!inputType)
    return failure();

  if (inputType.getElementType() != result().getType()) {
    return emitOpError() << "Type mismatch: Extracting from a " << inputType
                         << " should produce a " << inputType.getElementType()
                         << " but this op returns a " << result().getType();
  }

  return success();
}
