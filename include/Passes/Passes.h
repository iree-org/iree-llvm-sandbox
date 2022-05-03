//===- Passes.h - Linalg pass entry points ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_PASSES_H
#define IREE_LLVM_SANDBOX_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to drive tile + fuse transformations.
std::unique_ptr<OperationPass<FuncOp>> createLinalgFusePass();

/// Creates a pass to driver fuse output into reduction transformations.
std::unique_ptr<OperationPass<FuncOp>>
createLinalgFuseOutputIntoReductionPass();

/// Create a pass to drive the unrolling of a single vector op.
std::unique_ptr<OperationPass<FuncOp>> createUnrollOneVectorOpPass();

//===----------------------------------------------------------------------===//
// Transforms
//===----------------------------------------------------------------------===//

/// Add staged lowering of vector ops. `passManager` is expected to be a
/// `func.func` op pass manager.
void addLowerToVectorTransforms(OpPassManager &passManager);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Passes/Passes.h.inc"

} // namespace mlir

#endif // IREE_LLVM_SANDBOX_PASSES_H
