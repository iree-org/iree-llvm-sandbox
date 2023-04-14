//===-- IndexingTypes.td - Indexing dialect types --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_DIALECT_INDEXING_IR_INDEXINGTYPES
#define INDEXING_DIALECT_INDEXING_IR_INDEXINGTYPES

include "indexing/Dialect/Indexing/IR/IndexingDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Indexing type definitions
//===----------------------------------------------------------------------===//

class Indexing_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Indexing_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Indexing_CustomType : Indexing_Type<"Custom", "custom"> {
    let summary = "Indexing custom type";
    let description = "Custom type in indexing dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // INDEXING_DIALECT_INDEXING_IR_INDEXINGTYPES
