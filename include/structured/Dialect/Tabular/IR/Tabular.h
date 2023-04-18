//===-- Tabular.h - Tabular dialect -----------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_TABULAR_IR_TABULAR_H
#define ITERATORS_DIALECT_TABULAR_IR_TABULAR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "structured/Dialect/Tabular/IR/TabularOpsDialect.h.inc"

namespace mlir {
namespace tabular {
#include "structured/Dialect/Tabular/IR/TabularOpInterfaces.h.inc"
#include "structured/Dialect/Tabular/IR/TabularTypeInterfaces.h.inc"
} // namespace tabular
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "structured/Dialect/Tabular/IR/TabularOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "structured/Dialect/Tabular/IR/TabularOps.h.inc"

#endif // ITERATORS_DIALECT_TABULAR_IR_TABULAR_H
