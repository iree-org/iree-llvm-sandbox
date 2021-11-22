//===-- LinalgExtDialect.h - Linalg Extension dialect ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_LINALGEXT_LINALGEXTBASE_H
#define DIALECTS_LINALGEXT_LINALGEXTBASE_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

// clang-format off: must be included after all LLVM/MLIR headers
#include "Dialects/LinalgExt/LinalgExtOpsDialect.h.inc"  // IWYU pragma: keep
// clang-format on

#endif // DIALECTS_LINALGEXT_LINALGEXTBASE_H
