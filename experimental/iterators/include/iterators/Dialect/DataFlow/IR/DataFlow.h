//===-- DataFlowDialect.h - DataFlow dialect --------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOW_H
#define ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOW_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "iterators/Dialect/DataFlow/IR/DataFlowOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/DataFlow/IR/DataFlowOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "iterators/Dialect/DataFlow/IR/DataFlowOps.h.inc"

#endif // ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOW_H
