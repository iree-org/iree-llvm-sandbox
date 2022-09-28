//===-- Tabular.cpp - Tabular dialect ---------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tabular/IR/Tabular.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Tabular dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tabular/IR/TabularOpsDialect.cpp.inc"

void TabularDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/Tabular/IR/TabularOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/Tabular/IR/TabularOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tabular interfaces
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tabular/IR/TabularOpInterfaces.cpp.inc"
#include "iterators/Dialect/Tabular/IR/TabularTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Tabular operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Tabular types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOpsTypes.cpp.inc"
