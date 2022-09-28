//===-- TabularDialect.h - Tabular dialect ----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TABULAR_DIALECT_TABULAR_IR_TABULAR_H
#define TABULAR_DIALECT_TABULAR_IR_TABULAR_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "iterators/Dialect/Tabular/IR/TabularOpsDialect.h.inc"

namespace mlir {
namespace iterators {
#include "iterators/Dialect/Tabular/IR/TabularOpInterfaces.h.inc"
#include "iterators/Dialect/Tabular/IR/TabularTypeInterfaces.h.inc"
} // namespace iterators
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOps.h.inc"

#endif // TABULAR_DIALECT_TABULAR_IR_TABULAR_H
