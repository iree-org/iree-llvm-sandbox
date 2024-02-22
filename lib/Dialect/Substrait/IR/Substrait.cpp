//===-- Substrait.cpp - Substrait dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Substrait/IR/Substrait.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::substrait;

//===----------------------------------------------------------------------===//
// Substrait dialect
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Substrait/IR/SubstraitOpsDialect.cpp.inc"

void SubstraitDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "structured/Dialect/Substrait/IR/SubstraitOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "structured/Dialect/Substrait/IR/SubstraitOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Substrait operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "structured/Dialect/Substrait/IR/SubstraitOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Substrait types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "structured/Dialect/Substrait/IR/SubstraitOpsTypes.cpp.inc"
