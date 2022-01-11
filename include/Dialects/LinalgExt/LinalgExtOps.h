//===-- LinalgExtOps.h - Linalg Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_LINALGEXT_LINALGEXTOPS_H
#define DIALECTS_LINALGEXT_LINALGEXTOPS_H

#include "Dialects/LinalgExt/LinalgExtDialect.h"
#include "Dialects/LinalgExt/LinalgExtInterfaces.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Support/LLVM.h"

#define GET_OP_CLASSES
#include "Dialects/LinalgExt/LinalgExtOps.h.inc"

#endif // DIALECTS_LINALGEXT_LINALGEXTOPS_H
