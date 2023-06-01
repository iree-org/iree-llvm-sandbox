//===-- Passes.h - Indexing Pass Construction and Registration -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef STRUCTURED_DIALECT_INDEXING_TRANSFORMS_PASSES_H
#define STRUCTURED_DIALECT_INDEXING_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

/// Generate pass declarations.
#define GEN_PASS_DECL
#include "structured/Dialect/Indexing/Transforms/Passes.h.inc"

std::unique_ptr<Pass> createGatherToExtractSlicePass();
std::unique_ptr<Pass> createScatterToInsertSlicePass();
std::unique_ptr<Pass> createMungeCallingConventionsPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "structured/Dialect/Indexing/Transforms/Passes.h.inc"

} // namespace mlir

#endif // STRUCTURED_DIALECT_INDEXING_TRANSFORMS_PASSES_H
