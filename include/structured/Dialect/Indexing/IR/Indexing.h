//===-- Indexing.h - Indexing dialect ---------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_DIALECT_INDEXING_IR_INDEXING_H
#define INDEXING_DIALECT_INDEXING_IR_INDEXING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "structured/Dialect/Indexing/IR/IndexingOpsDialect.h.inc"

namespace mlir {
namespace indexing {} // namespace indexing
} // namespace mlir

#define GET_TYPEDEF_CLASSES

#include "structured/Dialect/Indexing/IR/IndexingOpsTypes.h.inc"

#define GET_OP_CLASSES

#include "structured/Dialect/Indexing/IR/IndexingOps.h.inc"

#endif // INDEXING_DIALECT_INDEXING_IR_INDEXING_H
