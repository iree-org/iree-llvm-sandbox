//===-- LinalgTransformOps.h - Linalg Transform dialect ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H
#define MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H

#include "Dialects/LinalgTransform/TrackingListener.h"
#include "Dialects/LinalgTransform/TransformOpInterface.h"
#include "TrackingListener.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace scf {
class ForOp;
} // namespace scf
} // namespace mlir

#include "Dialects/LinalgTransform/LinalgTransformOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "Dialects/LinalgTransform/LinalgTransformOps.h.inc"

#endif // MLIR_DIALECT_LINALG_IR_LINALGTRANSFORMOPS_H
