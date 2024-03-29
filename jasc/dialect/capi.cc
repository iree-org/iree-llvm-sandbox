//===-- capi.cc - C-API for the Jasc dialect --------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#include "capi.h"
#include "mlir/CAPI/Registration.h"
#include "dialect.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Jasc, jasc, jasc::JascDialect)
