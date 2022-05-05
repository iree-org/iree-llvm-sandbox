//===- IteratorsPasses.cpp - C API for Iterators Dialect Passes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Conversion/Passes.h"
#include "mlir-c/Pass.h"    // IWYU pragma: keep
#include "mlir/CAPI/Pass.h" // IWYU pragma: keep

using namespace mlir;

// Must include the declarations as they carry important visibility attributes.
#include "iterators/Conversion/Passes.capi.h.inc" // IWYU pragma: keep

#ifdef __cplusplus
extern "C" {
#endif

#include "iterators/Conversion/Passes.capi.cpp.inc" // IWYU pragma: keep

#ifdef __cplusplus
}
#endif
