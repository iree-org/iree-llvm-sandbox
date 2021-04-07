//===- LinalgTensorPasses.py.cpp - Pybind module for LinalgTensor passes --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include "Passes.capi.h.inc"

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

PYBIND11_MODULE(_mlirLinalgTensorPasses, m) {
  m.doc() = "MLIR Linalg Tensor Dialect Passes";

  // Register all Linalg passes on load.
  mlirRegisterLinalgTensorCodegenStrategyPass();
}
