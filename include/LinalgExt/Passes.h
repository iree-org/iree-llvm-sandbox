//===-- Passes.h - Linalg Ext transformation passes --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RUNNERS_LINALGEXT_PASSES_H_
#define RUNNERS_LINALGEXT_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linalg_ext {

/// Creates a pass to drive tiling of LinalgExt operations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgExtTilingPass(
    ArrayRef<int64_t> tileSizes = {});

std::unique_ptr<OperationPass<FuncOp>> createLinalgExtToSCFPass();

#define GEN_PASS_REGISTRATION
#include "LinalgExt/Passes.h.inc"

}  // namespace linalg_ext
}  // namespace mlir

#endif  // RUNNERS_LINALGEXT_PASSES_H_
