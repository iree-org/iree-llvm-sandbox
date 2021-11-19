//===-- VectorExtOps.h - Vector Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_VECTOREXT_VECTOREXTOPS_H
#define RUNNERS_VECTOREXT_VECTOREXTOPS_H

#include "VectorExt/VectorExtDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "VectorExt/VectorExtOps.h.inc"

#endif  // RUNNERS_VECTOREXT_VECTOREXTOPS_H
