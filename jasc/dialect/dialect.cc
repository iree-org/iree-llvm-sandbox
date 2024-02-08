//===-- dialect.cc - Jasc dialect implementation ----------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "ops.h"

// Include code generated from dialect.td.
#include "dialect/dialect.cc.inc"

namespace jasc {

void JascDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "dialect/ops.cc.inc"
      >();
}

}  // namespace jasc
