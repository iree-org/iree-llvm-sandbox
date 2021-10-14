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

#ifndef RUNNERS_PASSES_H
#define RUNNERS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

/// Creates a pass to drive transformations on Linalg on tensors.
std::unique_ptr<OperationPass<ModuleOp>> createLinalgTensorCodegenDriverPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

}  // namespace mlir

#endif  // RUNNERS_PASSES_H
