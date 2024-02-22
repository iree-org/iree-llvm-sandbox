//===-- SubstraitTypes.td - Substrait dialect types --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES

include "structured/Dialect/Substrait/IR/SubstraitDialect.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

// Base class for Substrait dialect types.
class Substrait_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Substrait_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES
