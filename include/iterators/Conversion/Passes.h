//===- Passes.h - Conversion Pass Construction and Registration -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_CONVERSION_PASSES_H
#define ITERATORS_CONVERSION_PASSES_H

#include "iterators/Conversion/IteratorsToLLVM/IteratorsToLLVM.h"
#include "iterators/Conversion/StatesToLLVM/StatesToLLVM.h"
#include "iterators/Conversion/TabularToLLVM/TabularToLLVM.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "iterators/Conversion/Passes.h.inc"

} // namespace mlir

#endif // ITERATORS_CONVERSION_PASSES_H
