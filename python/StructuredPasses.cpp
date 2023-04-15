//===- IteratorsPasses.cpp - Pybind module for passes on Iterators --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured-c/Passes.h"

#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirStructuredPasses, m) {
  m.doc() = "MLIR Structured Dialect Passes";

  // Register all Structured passes on load.
  mlirRegisterStructuredConversionPasses();
  mlirRegisterIteratorsPasses();
  mlirRegisterTuplePasses();
}
