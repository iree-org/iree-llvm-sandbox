//===-- Substrait.h - Substrait dialect -------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_SUBSTRAIT_IR_SUBSTRAIT_H
#define STRUCTURED_DIALECT_SUBSTRAIT_IR_SUBSTRAIT_H

#include "mlir/Dialect/Func/IR/FuncOps.h"         // IWYU: keep
#include "mlir/IR/Dialect.h"                      // IWYU: keep
#include "mlir/IR/OpImplementation.h"             // IWYU: keep
#include "mlir/IR/SymbolTable.h"                  // IWYU: keep
#include "mlir/Interfaces/InferTypeOpInterface.h" // IWYU: keep

#include "structured/Dialect/Substrait/IR/SubstraitOpsDialect.h.inc" // IWYU: export

#define GET_TYPEDEF_CLASSES
#include "structured/Dialect/Substrait/IR/SubstraitOpsTypes.h.inc" // IWYU: export

#define GET_OP_CLASSES
#include "structured/Dialect/Substrait/IR/SubstraitOps.h.inc" // IWYU: export

#endif // STRUCTURED_DIALECT_SUBSTRAIT_IR_SUBSTRAIT_H
