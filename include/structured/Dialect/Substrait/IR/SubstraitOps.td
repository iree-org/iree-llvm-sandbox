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
include "structured/Dialect/Substrait/IR/SubstraitTypes.td"
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

def Substrait_PlanRelOp : Substrait_Op<"relation", [
    HasParent<"::mlir::substrait::PlanOp">
  ]> {
  let summary = "Represents the result of a Substrait plan";
  let assemblyFormat = "attr-dict";
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITOPS
