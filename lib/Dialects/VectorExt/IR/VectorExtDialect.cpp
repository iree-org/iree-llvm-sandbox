//===-- VectorExtDialect.cpp - Vector Extension dialect -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorExtDialect.h"

#include "Dialects/VectorExt/VectorExtOps.h"
#include "Dialects/VectorExt/VectorExtOpsDialect.cpp.inc"

namespace mlir {
namespace vector_ext {

void VectorExtDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "Dialects/VectorExt/VectorExtOps.cpp.inc"
      >();
}

} // namespace vector_ext
} // namespace mlir
