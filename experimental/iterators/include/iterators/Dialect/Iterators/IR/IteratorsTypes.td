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

//===----------------------------------------------------------------------===//
// State of low-level iterators
//===----------------------------------------------------------------------===//

class Iterators_Iterator<string name, string typeMnemonic>
    : Iterators_Type<name, typeMnemonic, [DeclareTypeInterfaceMethods<Iterators_IteratorInterface>]> {
  let summary = "State of a Volcano-style iterator";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
}

def Iterators_SampleInputState : Iterators_Iterator<"SampleInputState", "sampleinputstate">;

def Iterators_ReduceState : Iterators_Iterator<"ReduceState", "reducestate">;

def IsIteratorTypePred : CPred<"$_self.isa<::mlir::iterators::IteratorInterface>()">;
def AnyIteratorState : Type<IsIteratorTypePred, "iteratorstate", "::mlir::Type">;

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
