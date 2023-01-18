//===-- VectorExtOps.h - Vector Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_VECTOREXT_VECTOREXTOPS_H
#define DIALECTS_VECTOREXT_VECTOREXTOPS_H

#include "Dialect/VectorExt/VectorExtDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace vector_ext {

void buildTerminatedBody(OpBuilder &builder, Location loc);

} // namespace vector_ext
} // namespace mlir

#define GET_OP_CLASSES
#include "Dialect/VectorExt/VectorExtOps.h.inc"

#endif // DIALECTS_VECTOREXT_VECTOREXTOPS_H
