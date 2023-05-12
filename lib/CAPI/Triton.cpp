//===-- Triton.cpp - C API for Triton Passes --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "structured-c/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"
#include "structured/Conversion/Passes.h"

using namespace mlir;
using namespace mlir::triton;

// Must include the declarations as they carry important visibility attributes.
#include "structured/Conversion/TritonConversions.capi.h.inc"
#include "structured/Dialect/Triton/Transforms/Passes.capi.h.inc"
#include "structured/Dialect/TritonGPU/Transforms/Passes.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

#include "structured/Conversion/TritonConversions.capi.cpp.inc"
#include "structured/Dialect/Triton/Transforms/Passes.capi.cpp.inc"
#include "structured/Dialect/TritonGPU/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
