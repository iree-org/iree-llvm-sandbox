//===- IteratorsTypes.td - Iterators dialect types ---------*- tablegen -*-===//

#ifndef DIALECTS_ITERATORS_ITERATORSTYPES
#define DIALECTS_ITERATORS_ITERATORSTYPES

include "mlir/IR/AttrTypeBase.td"
include "Dialects/Iterators/IteratorsDialect.td"

// Base class for Iterators dialect types.
class Iterators_Type<string name, string typeMnemonic>
    : TypeDef<Iterators_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def Iterators_Iterator : Iterators_Type<"Iterator", "iterator"> {
  let summary = "Basic iterator type";
}

#endif // DIALECTS_ITERATORS_ITERATORSTYPES
