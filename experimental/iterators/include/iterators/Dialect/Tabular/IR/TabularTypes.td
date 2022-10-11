//===-- TabularTypes.td - Tabular dialect types ------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TABULAR_DIALECT_TABULAR_IR_TABULARTYPES
#define TABULAR_DIALECT_TABULAR_IR_TABULARTYPES

include "iterators/Dialect/Tabular/IR/TabularDialect.td"
include "iterators/Dialect/Tabular/IR/TabularInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

// Base class for Tabular dialect types.
class Tabular_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Tabular_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // TABULAR_DIALECT_TABULAR_IR_TABULARTYPES
