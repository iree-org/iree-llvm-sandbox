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

/// Creates a pass to drive tiling of LinalgExt operations to scf::ForOp.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgExtTilingPass(ArrayRef<int64_t> tileSizes = {});

/// Creates a pass to drive tiling of LinalgExt operations to
/// linalg_ext::TileOp.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgExtTilingToTileOpPass(ArrayRef<int64_t> tileSizes = {});

std::unique_ptr<OperationPass<FuncOp>> createInParallelToAsyncPass();
std::unique_ptr<OperationPass<FuncOp>> createInParallelToSequentialForPass();
std::unique_ptr<OperationPass<FuncOp>> createTileToSequentialForPass();
std::unique_ptr<OperationPass<FuncOp>> createTileToInParallelPass();

void registerTilingInterfaceExternalModels(DialectRegistry &registry);

#define GEN_PASS_REGISTRATION
#include "Dialects/LinalgExt/Passes.h.inc"

} // namespace linalg_ext
} // namespace mlir

#endif // DIALECTS_LINALGEXT_PASSES_H_
