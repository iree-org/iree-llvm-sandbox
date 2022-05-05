//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators-c/Dialects.h"

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

//===----------------------------------------------------------------------===//
// Iterators dialect and types
//===----------------------------------------------------------------------===//

using namespace mlir::iterators;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Iterators, iterators, IteratorsDialect)

bool mlirTypeIsAIteratorsStreamType(MlirType type) {
  return unwrap(type).isa<StreamType>();
}

MlirType mlirIteratorsStreamTypeGet(MlirContext context, MlirType elementType) {
  return wrap(StreamType::get(unwrap(context), unwrap(elementType)));
}
