//===-- LinalgExtDialect.h - Linalg Extension dialect ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "include/LinalgExt/LinalgExtDialect.h"

#include "include/LinalgExt/LinalgExtDialect.cpp.inc"
#include "include/LinalgExt/LinalgExtOps.h"

namespace mlir {
namespace linalg_ext {

void LinalgExtDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "include/LinalgExt/LinalgExtOps.cpp.inc"
      >();
}

}  // namespace linalg_ext
}  // namespace mlir
