//===-- TabularOps.td - Tabular operations definitions -----*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TABULAR_DIALECT_TABULAR_IR_TABULAROPS
#define TABULAR_DIALECT_TABULAR_IR_TABULAROPS

include "iterators/Dialect/Tabular/IR/TabularDialect.td"
include "iterators/Dialect/Tabular/IR/TabularTypes.td"
include "mlir/IR/OpBase.td"

class Tabular_Op<string mnemonic, list<Trait> traits = []> :
  Op<Tabular_Dialect, mnemonic, traits> {
}

#endif // TABULAR_DIALECT_TABULAR_IR_TABULAROPS
