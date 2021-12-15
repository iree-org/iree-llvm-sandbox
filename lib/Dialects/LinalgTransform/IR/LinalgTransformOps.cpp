//===-- LinalgTransformOps.cpp - Linalg Transform dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/LinalgTransformOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Dialects/LinalgTransform/LinalgTransformOpsDialect.cpp.inc"

void mlir::linalg::transform::LinalgTransformDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialects/LinalgTransform/LinalgTransformOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "Dialects/LinalgTransform/LinalgTransformOps.cpp.inc"
