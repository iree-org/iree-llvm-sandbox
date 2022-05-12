//===-- Iterators.cpp - Iterators dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/Iterators.h"

#include <assert.h>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectImplementation.h" // IWYU pragma: keep
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"  // IWYU pragma: keep
#include "llvm/ADT/TypeSwitch.h" // IWYU pragma: keep

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Iterators dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.cpp.inc" // IWYU pragma: keep

void IteratorsDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc" // IWYU pragma: keep
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc" // IWYU pragma: keep
      >();
}

//===----------------------------------------------------------------------===//
// Iterators interfaces
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/IR/IteratorsOpInterfaces.cpp.inc" // IWYU pragma: keep
#include "iterators/Dialect/Iterators/IR/IteratorsTypeInterfaces.cpp.inc" // IWYU pragma: keep

//===----------------------------------------------------------------------===//
// Iterators operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.cpp.inc" // IWYU pragma: keep

LogicalResult CreateSampleInputStateOp::verify() {
  IteratorInterface iteratorType =
      createdState().getType().dyn_cast<IteratorInterface>();
  assert(iteratorType);

  TupleType tupleType =
      TupleType::get(getContext(), {IntegerType::get(getContext(), 32)});
  if (iteratorType.getElementType() != tupleType) {
    return emitOpError() << "Type mismatch: Sample input iterator (currently) "
                            "has to return elements of type 'tuple<i32>'";
  }

  return success();
}

LogicalResult CreateReduceStateOp::verify() {
  IteratorInterface iteratorType =
      createdState().getType().dyn_cast<IteratorInterface>();
  assert(iteratorType);

  IteratorInterface upstreamIteratorType =
      upstreamState().getType().dyn_cast<IteratorInterface>();
  assert(upstreamIteratorType);

  if (iteratorType.getElementType() != upstreamIteratorType.getElementType()) {
    return emitOpError() << "Type mismatch: Upstream iterator of reduce "
                            "iterator must produce elements of type "
                         << iteratorType.getElementType()
                         << " but produces elements of type "
                         << upstreamIteratorType.getElementType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Iterators types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.cpp.inc" // IWYU pragma: keep
