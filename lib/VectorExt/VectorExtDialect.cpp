//===-- VectorExtDialect.h - Vector Extension dialect ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VectorExt/VectorExtDialect.h"

#include "VectorExt/VectorExtOps.h"
#include "VectorExt/VectorExtOpsDialect.cpp.inc"

namespace mlir {
namespace vector_ext {

void VectorExtDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "VectorExt/VectorExtOps.cpp.inc"
      >();
}

}  // namespace vector_ext
}  // namespace mlir
