//===- IteratorsTypes.td - Iterators dialect types ---------*- tablegen -*-===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES

include "iterators/Dialect/Iterators/IR/IteratorsDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// Base class for Iterators dialect types.
class Iterators_Type<string name, string typeMnemonic>
    : TypeDef<Iterators_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Iterators_Iterator : Iterators_Type<"Iterator", "iterator"> {
  let summary = "Basic iterator type";
  let parameters = (ins "Type":$typesTuple);
  let assemblyFormat = "`<` qualified($typesTuple) `>`";
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSTYPES
