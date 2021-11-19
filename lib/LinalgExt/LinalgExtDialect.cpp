//===-- LinalgExtDialect.h - Linalg Extension dialect ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinalgExt/LinalgExtDialect.h"

#include "LinalgExt/LinalgExtOps.h"
#include "LinalgExt/LinalgExtOpsDialect.cpp.inc"

namespace mlir {
namespace linalg_ext {

void LinalgExtDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "LinalgExt/LinalgExtOps.cpp.inc"
      >();
}

}  // namespace linalg_ext
}  // namespace mlir
