//===- Passes.h - Substrait pass declarations -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES_H_
#define STRUCTURED_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace substrait {

#define GEN_PASS_DECL
#include "structured/Dialect/Substrait/Transforms/Passes.h.inc"

/// Create a pass to eliminate duplicate fields in `emit` ops.
std::unique_ptr<Pass> createEmitDeduplicationPass();

/// Add patterns that eliminate duplicate fields in `emit` ops.
void populateEmitDeduplicationPatterns(RewritePatternSet &patterns);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "structured/Dialect/Substrait/Transforms/Passes.h.inc"

} // namespace substrait
} // namespace mlir

#endif // STRUCTURED_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES_H_
