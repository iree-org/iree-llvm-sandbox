//===-- Passes.h - Linalg Ext transformation passes --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef DIALECTS_LINALGEXT_PASSES_H_
#define DIALECTS_LINALGEXT_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class DialectRegistry;

namespace linalg_ext {

void registerTilingInterfaceExternalModels(DialectRegistry &registry);

} // namespace linalg_ext
} // namespace mlir

#endif // DIALECTS_LINALGEXT_PASSES_H_
