//===-- ops.h - Ops of the Jasc dialect -------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef THIRD_PARTY_MLIR_EDGE_JASC_DIALECT_OPS_H_
#define THIRD_PARTY_MLIR_EDGE_JASC_DIALECT_OPS_H_

#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES
#include "dialect/ops.h.inc"

#endif  // THIRD_PARTY_MLIR_EDGE_JASC_DIALECT_OPS_H_
