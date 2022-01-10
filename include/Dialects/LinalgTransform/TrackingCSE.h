//===-- TrackingCSE.h - Special common subexpr elimination ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H
#define MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H

#include "Dialects/LinalgTransform/TransformOpMapping.h"
#include "mlir/Support/LLVM.h"

namespace mlir {

class Operation;
class DominanceInfo;
class Value;

void eliminateCommonSubexpressionsWithTrackedOps(
    Operation *root, TransformOpMapping &trackedOps,
    DominanceInfo *domInfo = nullptr);
} // namespace mlir

#endif // MLIR_DIALECT_LINALG_TRANSFORMS_TRACKINGCSE_H
