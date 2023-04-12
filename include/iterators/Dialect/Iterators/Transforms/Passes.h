//===- Passes.h - Transform Pass Construction and Registration --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_TRANSFORMS_PASSES_H
#define ITERATORS_DIALECT_ITERATORS_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// Construction
//===----------------------------------------------------------------------===//

/// Generate pass declarations.
#define GEN_PASS_DECL
#include "iterators/Dialect/Iterators/Transforms/Passes.h.inc"

/// Creates a pass that decomposes iterator states into individual values.
std::unique_ptr<Pass> createDecomposeIteratorStatesPass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "iterators/Dialect/Iterators/Transforms/Passes.h.inc"

} // namespace mlir

#endif // ITERATORS_DIALECT_ITERATORS_TRANSFORMS_PASSES_H
