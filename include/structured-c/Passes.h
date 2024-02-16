//===-- Passes.h - CAPI for passes --------------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_C_PASSES_H
#define STRUCTURED_C_PASSES_H

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Structured passes
//===----------------------------------------------------------------------===//

#include "structured/Conversion/Passes.capi.h.inc" // IWYU pragma: export

#include "structured/Dialect/Iterators/Transforms/Passes.capi.h.inc" // IWYU pragma: export

#include "structured/Dialect/Tuple/Transforms/Passes.capi.h.inc" // IWYU pragma: export

#ifdef __cplusplus
}
#endif

#endif // STRUCTURED_C_PASSES_H
