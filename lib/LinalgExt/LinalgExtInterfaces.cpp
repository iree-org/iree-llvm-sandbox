//===-- LinalgExtInterface.h - Linalg Extension interface --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LinalgExt/LinalgExtInterfaces.h"

#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::linalg_ext;

#include "LinalgExt/LinalgExtInterfaces.cpp.inc"  // IWYU pragma: export
