//===-- Tuple.h - Tuple dialect header file ---------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_TUPLE_IR_TUPLE_H
#define STRUCTURED_DIALECT_TUPLE_IR_TUPLE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"

#include "structured/Dialect/Tuple/IR/TupleOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "structured/Dialect/Tuple/IR/TupleOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "structured/Dialect/Tuple/IR/TupleOps.h.inc"

#endif // STRUCTURED_DIALECT_TUPLE_IR_TUPLE_H
