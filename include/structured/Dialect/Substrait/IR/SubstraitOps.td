//===-- SubstraitOps.td - Substrait operations definitions -*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITOPS
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITOPS

include "structured/Dialect/Substrait/IR/SubstraitDialect.td"
include "structured/Dialect/Substrait/IR/SubstraitInterfaces.td"
include "structured/Dialect/Substrait/IR/SubstraitTypes.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"

class Substrait_Op<string mnemonic, list<Trait> traits = []> :
  Op<Substrait_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Constraints
//===----------------------------------------------------------------------===//

class RegionOf<Constraint condition> : Region<
  Concat<"::llvm::all_of($_self.getOps(), [](::mlir::Operation &op) { "
         "return ",
         SubstLeaves<"$_self", "op", condition.predicate>,
         "; })">,
  "region where each child op is " # condition.summary>;

class IsOp<string opType> : Constraint<
  CPred<"::llvm::isa<" # opType # ">($_self)">,
  "op of type '" # opType # "'">;

class AnyOf<list<Constraint> opTypes, string summary = ""> : Constraint<
  Or<!foreach(opType, opTypes, opType.predicate)>,
     !if(!eq(summary, ""),
        !interleave(!foreach(t, opTypes, t.summary), " or "),
        summary)>;

def StringArrayAttr :
    TypedArrayAttrBase<Builtin_StringAttr,
                       "array attribute of string attributes"> {
  let storageType = [{ ::mlir::ArrayAttr }];
}

//===----------------------------------------------------------------------===//
// Plan
//===----------------------------------------------------------------------===//
// The definitions in this section are related to the top-level `Plan` message.
// See https://substrait.io/serialization/binary_serialization/ and
// https://github.com/substrait-io/substrait/blob/main/proto/substrait/plan.proto.
//===----------------------------------------------------------------------===//

def PlanBodyOp : AnyOf<[
    IsOp<"::mlir::substrait::PlanRelOp">
  ]>;

def Substrait_PlanOp : Substrait_Op<"plan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    NoTerminator, NoRegionArguments, SingleBlock
  ]> {
  let summary = "Represents a Substrait plan";
  let description = [{
    This op represents the `Plan` message type of Substrait. It carries the
    version information inline as attributes, so it also subsumes the `Version`
    message type. The body of the op consists of the `relation`s and (once
    implemented) the extensions and types as well as their URLs defined in the
    plan.
  }];
  let arguments = (ins
    UI32Attr:$major_number,
    UI32Attr:$minor_number,
    UI32Attr:$patch_number,
    DefaultValuedAttr<StrAttr, "\"\"">:$git_hash,
    DefaultValuedAttr<StrAttr, "\"\"">:$producer
  );
  let regions = (region RegionOf<PlanBodyOp>:$body);
  let assemblyFormat = [{
    `version` $major_number `:` $minor_number `:` $patch_number
    (`git_hash` $git_hash^)? (`producer` $producer^)?
    attr-dict-with-keyword $body
  }];
  let builders = [
      OpBuilder<(ins "uint32_t":$major, "uint32_t":$minor, "uint32_t":$patch), [{
        build($_builder, $_state, major, minor, patch,
              StringAttr(), StringAttr());
      }]>
    ];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    ::llvm::StringRef $cppClass::getDefaultDialect() {
      return SubstraitDialect::getDialectNamespace();
    }
  }];
}

def RelationBodyOp : AnyOf<[
    IsOp<"::mlir::substrait::RelOpInterface">,
    IsOp<"::mlir::substrait::YieldOp">
  ]>;

def Substrait_PlanRelOp : Substrait_Op<"relation", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    HasParent<"::mlir::substrait::PlanOp">,
    SingleBlockImplicitTerminator<"::mlir::substrait::YieldOp">,
    NoRegionArguments
  ]> {
  let summary = "Represents a query tree in a Substrait plan";
  let description = [{
    Represents a `PlanRel` message, which is used in the `relations` field of
    the `Plan` message. The body of this op contains various `RelOpInterface`
    ops (corresponding to the `Rel` message type) producing SSA values and the
    one being yielded reprents the root of the query tree that this op contains.
  }];
  let regions = (region RegionOf<RelationBodyOp>:$body);
  let assemblyFormat = "attr-dict-with-keyword $body";
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    ::llvm::StringRef $cppClass::getDefaultDialect() {
      return SubstraitDialect::getDialectNamespace();
    }
  }];
}

def Substrait_YieldOp : Substrait_Op<"yield", [
    Terminator, HasParent<"::mlir::substrait::PlanRelOp">
  ]> {
  let summary = "Yields the result of a `PlanRelOp`";
  let arguments = (ins AnyType:$value);
  let assemblyFormat = "$value attr-dict `:` type($value)";
  let builders = [OpBuilder<(ins), [{ /* do nothing */ }]>];
}

//===----------------------------------------------------------------------===//
// Relations
//===----------------------------------------------------------------------===//
// The definitions in this section are related to the various `Rel` message
// types. See https://substrait.io/relations/basics/ and
// https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto.
//===----------------------------------------------------------------------===//

/// Attaches all traits that ops representing a `Rel` message type should have.
class Substrait_RelOp<string mnemonic, list<Trait> traits = []> :
  Substrait_Op<mnemonic, traits # [
    Substrait_RelOpInterface,
    PredOpTrait<"result must be extactly one Relation",
      And<[
        CPred<"$_op.getNumResults() == 1">,
        SubstLeaves<"$_self", "$_op.getResult(0).getType()",
          Substrait_Relation.predicate>
      ]>>
  ]>;

def Substrait_NamedTableOp : Substrait_RelOp<"named_table", [
  ]> {
  let summary = "Read operation of a named table";
  let description = [{
    Represents a `NamedTable` message together with the `ReadRel` and `Rel`
    messages that contain it.

    Example:

    ```mlir
    %0 = named_table @t1 as ["a"] : tuple<si32>
    ```
  }];
  // TODO(ingomueller): Maybe the result names should be part of a to-be-created
  //                    `NamedStruct` type?
  let arguments = (ins
    SymbolRefAttr:$table_name,
    StringArrayAttr:$field_names
  );
  let results = (outs Substrait_Relation:$result);
  let assemblyFormat = [{
    $table_name `as` $field_names attr-dict `:` type($result)
  }];
  let hasVerifier = true;
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITOPS
