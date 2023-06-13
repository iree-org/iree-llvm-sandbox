//===- Passes.h - Conversion Pass Construction and Registration -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_CONVERSION_PASSES_H
#define STRUCTURED_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include "structured/Conversion/IteratorsToLLVM/IteratorsToLLVM.h"
#include "structured/Conversion/StatesToLLVM/StatesToLLVM.h"
#include "structured/Conversion/TabularToLLVM/TabularToLLVM.h"
#include "structured/Conversion/TritonFuncToFunc/TritonFuncToFunc.h"
#include "structured/Conversion/TritonToLLVM/TritonToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TritonGPUToLLVMPass.h"
#include "triton/Conversion/TritonToTritonGPU/TritonToTritonGPUPass.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"

namespace mlir {

/// Generate the code for registering conversion passes.
#define GEN_PASS_REGISTRATION
#include "structured/Conversion/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "structured/Conversion/TritonConversions.h.inc"

} // namespace mlir

#endif // STRUCTURED_CONVERSION_PASSES_H
