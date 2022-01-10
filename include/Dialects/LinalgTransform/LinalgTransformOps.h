//===-- LinalgTransformOps.h - Linalg Transform dialect ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/OpDefinition.h"

#include "Dialects/LinalgTransform/LinalgTransformOpsDialect.h.inc"
#include <mlir/IR/BuiltinAttributes.h>

namespace mlir {
/// A trait for transform ops that can be targeted at either the result of a
/// matcher (identified by its symbol name) or at the result of another
/// transformation (identified by the value it produced).
template <typename ConcreteOp>
class TargetableTransformOpTrait
    : public OpTrait::TraitBase<ConcreteOp, TargetableTransformOpTrait> {
public:
  static LogicalResult verifyTrait(Operation *op) {
    Optional<SymbolRefAttr> matcher = cast<ConcreteOp>(op).targetMatcher();
    Value input = cast<ConcreteOp>(op).target();
    if (!((matcher.hasValue() && matcher.getValue() != nullptr) ^
          (input != nullptr)))
      return op->emitOpError()
             << "expects either an `op` operand or a `matcher` attribute";

    return success();
  }
};
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialects/LinalgTransform/LinalgTransformOps.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H
