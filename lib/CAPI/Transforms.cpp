//===- Transforms.cpp - C API for Transformations Passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// #include "iterators-c/Passes.h"
#include "iterators/Dialect/Iterators/Transforms/Passes.h"
#include "iterators/Dialect/Tuple/Transforms/Passes.h"
#include "mlir/CAPI/Pass.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

// Must include the declarations as they carry important visibility attributes.
#include "iterators/Dialect/Iterators/Transforms/Passes.capi.h.inc"

#include "iterators/Dialect/Tuple/Transforms/Passes.capi.h.inc"

#ifdef __cplusplus
extern "C" {
#endif

#include "iterators/Dialect/Iterators/Transforms/Passes.capi.cpp.inc"

#include "iterators/Dialect/Tuple/Transforms/Passes.capi.cpp.inc"

#ifdef __cplusplus
}
#endif
