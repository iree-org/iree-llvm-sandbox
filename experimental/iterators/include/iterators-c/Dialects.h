//===- Dialects.h - CAPI for dialects -----------------------------*- C -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_C_DIALECTS_H
#define ITERATORS_C_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
// Iterators dialect and types
//===----------------------------------------------------------------------===//

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Iterators, iterators);

MLIR_CAPI_EXPORTED
bool mlirTypeIsAIteratorsColumnarBatchType(MlirType type);

MLIR_CAPI_EXPORTED
MlirType mlirIteratorsColumnarBatchTypeGet(MlirContext context,
                                           MlirType elementType);

MLIR_CAPI_EXPORTED
bool mlirTypeIsAIteratorsStreamType(MlirType type);

MLIR_CAPI_EXPORTED
MlirType mlirIteratorsStreamTypeGet(MlirContext context, MlirType elementType);

#ifdef __cplusplus
}
#endif

#endif // ITERATORS_C_DIALECTS_H
