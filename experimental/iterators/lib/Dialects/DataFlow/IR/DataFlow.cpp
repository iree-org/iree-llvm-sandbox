//===-- DataFlow.cpp - DataFlow dialect -------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/DataFlow/IR/DataFlow.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// DataFlow dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/DataFlow/IR/DataFlowOpsDialect.cpp.inc"

void DataFlowDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/DataFlow/IR/DataFlowOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/DataFlow/IR/DataFlowOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// DataFlow operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iterators/Dialect/DataFlow/IR/DataFlowOps.cpp.inc"

//===----------------------------------------------------------------------===//
// DataFlow types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/DataFlow/IR/DataFlowOpsTypes.cpp.inc"
