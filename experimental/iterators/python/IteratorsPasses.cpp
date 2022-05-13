//===- IteratorsPasses.cpp - Pybind module for passes on Iterators --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators-c/Passes.h"

#include <pybind11/pybind11.h>

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirIteratorsPasses, m) {
  m.doc() = "MLIR Iterators Dialect Passes";

  // Register all Iterators passes on load.
  mlirRegisterIteratorsConversionPasses();
}
