//===- LinalgTensorPasses.py.cpp - C API for LinalgTensor Passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

// Must include the declarations as they carry important visibility attributes.
#include "Passes.capi.h.inc"

using namespace mlir;

#ifdef __cplusplus
extern "C" {
#endif

#include "Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
