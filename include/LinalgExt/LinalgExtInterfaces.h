//===-- Linalgextinterface.h - Linalg Extension interface --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_LINALGEXTINTERFACE_H
#define RUNNERS_LINALGEXTINTERFACE_H

#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace linalg_ext {

/// OpOperand vector that implicitly converts to a Value vector.
struct OpOperandVector : public SmallVector<OpOperand *> {
  operator SmallVector<Value>();
};

#include "include/LinalgExt/LinalgExtInterfaces.h.inc"  // IWYU pragma: export
}  // namespace linalg_ext
}  // namespace mlir

#endif  // RUNNERS_LINALGEXTINTERFACE_H
