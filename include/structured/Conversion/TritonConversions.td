//===-- Triton.td - Conversion passes from Triton ----------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_CONVERSION_TRITON
#define STRUCTURED_CONVERSION_TRITON

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Triton conversion passes
//===----------------------------------------------------------------------===//

include "triton/Conversion/TritonToTritonGPU/Passes.td"
include "triton/Conversion/TritonGPUToLLVM/Passes.td"

#endif // STRUCTURED_CONVERSION_TRITON
