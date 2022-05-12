//===-- IteratorsDialect.h - Iterators dialect ------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H

#include "mlir/IR/Dialect.h"                      // IWYU pragma: keep
#include "mlir/IR/OpImplementation.h"             // IWYU pragma: keep
#include "mlir/IR/SymbolTable.h"                  // IWYU pragma: keep
#include "mlir/Interfaces/InferTypeOpInterface.h" // IWYU pragma: keep

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.h.inc" // IWYU pragma: export

namespace mlir {
namespace iterators {
// IWYU pragma: begin_exports
#include "iterators/Dialect/Iterators/IR/IteratorsOpInterfaces.h.inc"
#include "iterators/Dialect/Iterators/IR/IteratorsTypeInterfaces.h.inc"
// IWYU pragma: end_exports
} // namespace iterators
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.h.inc" // IWYU pragma: export

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.h.inc" // IWYU pragma: export

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H
