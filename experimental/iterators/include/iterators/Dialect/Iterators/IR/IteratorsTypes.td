//===-- IteratorsTypes.td - Iterators dialect types --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES

include "iterators/Dialect/Iterators/IR/IteratorsDialect.td"
include "iterators/Dialect/Iterators/IR/IteratorsInterfaces.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// Base class for Iterators dialect types.
class Iterators_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Iterators_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Iterators_Stream : Iterators_Type<"Stream", "stream"> {
  let summary = "Stream of elements of the given type";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
