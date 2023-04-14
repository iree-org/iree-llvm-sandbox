//===- Dialects.cpp - CAPI for dialects -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "indexing-c/Dialects.h"

#include "indexing/Dialect/Indexing/IR/Indexing.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/StringRef.h"

//===----------------------------------------------------------------------===//
// Indexing dialect and types
//===----------------------------------------------------------------------===//

using namespace mlir;
using namespace mlir::indexing;

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Indexing, indexing, IndexingDialect)
