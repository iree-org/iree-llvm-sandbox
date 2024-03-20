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
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/OpBase.td"

// Base class for Substrait dialect types.
class Substrait_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Substrait_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

/// Currently supported atomic types. These correspond directly to the types in
/// https://github.com/substrait-io/substrait/blob/main/proto/substrait/type.proto.
// TODO(ingomueller): Add the other low-hanging fruits here.
def Substrait_AtomicTypes {
  list<Type> types = [
    SI32 // I32
  ];
}

/// One of the currently supported atomic types.
def Substrait_AtomicType : AnyTypeOf<Substrait_AtomicTypes.types>;

/// Placeholder for a proper relation type, the result of any `RelOpInterface`
/// op.
// TODO(ingomueller): Transform this into a proper relation type.
def Substrait_Relation : NestedTupleOf<Substrait_AtomicTypes.types>;

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITTYPES
