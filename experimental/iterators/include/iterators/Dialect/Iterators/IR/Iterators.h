//===-- IteratorsDialect.h - Iterators dialect ------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/SymbolTable.h"

#include "iterators/Dialect/Iterators/IR/IteratorsOpsDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "iterators/Dialect/Iterators/IR/IteratorsOps.h.inc"

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORS_H
