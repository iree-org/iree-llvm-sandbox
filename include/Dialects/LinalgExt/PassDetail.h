//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_LINALGEXT_PASSDETAIL_H_
#define DIALECTS_LINALGEXT_PASSDETAIL_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linalg_ext {

#define GEN_PASS_CLASSES
#include "Dialects/LinalgExt/Passes.h.inc"  // IWYU pragma: keep

}  // namespace linalg_ext
}  // namespace mlir

#endif // DIALECTS_LINALGEXT_PASSDETAIL_H_
