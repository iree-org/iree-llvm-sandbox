//===-- VectorExtDialect.cpp - Vector Extension dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/VectorExt/VectorExtDialect.h"

#include "Dialect/VectorExt/VectorExtOps.h"
#include "Dialect/VectorExt/VectorExtOpsDialect.cpp.inc"

namespace mlir {
namespace vector_ext {

void VectorExtDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "Dialect/VectorExt/VectorExtOps.cpp.inc"
      >();
}

} // namespace vector_ext
} // namespace mlir
