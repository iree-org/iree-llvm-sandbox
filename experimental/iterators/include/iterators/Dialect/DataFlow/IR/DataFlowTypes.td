//===-- DataFlowTypes.td - DataFlow dialect types ----------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWTYPES
#define ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWTYPES

include "iterators/Dialect/DataFlow/IR/DataFlowDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

// Base class for DataFlow dialect types.
class DataFlow_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<DataFlow_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DataFlow_Stream : DataFlow_Type<"Stream", "stream"> {
  let summary = "Stream of elements of the given type";
  let parameters = (ins "Type":$elementType);
  let assemblyFormat = "`<` qualified($elementType) `>`";
}

#endif // ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWTYPES
