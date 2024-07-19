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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonAttrConstraints.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

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
// Extensions
//===----------------------------------------------------------------------===//
// The definitions in this section are related to the extension messages.
// See https://substrait.io/serialization/binary_serialization/ and
// https://github.com/substrait-io/substrait/blob/main/proto/substrait/extensions/extensions.proto.
//===----------------------------------------------------------------------===//

def Substrait_ExtensionUriOp : Substrait_Op<"extension_uri", [
    Symbol
  ]> {
  let summary = "Declares a simple extension URI";
  let description = [{
    This op represents the `SimpleExtensionURI` message type of Substrait. It is
    a `Symbol` op, so it can be looked up in the symbol table of the plan it is
    contained in.

    Example code:

    ```mlir
    substrait.plan version 0 : 42 : 1 {
      extension_uri @uri at "http://some.url/with/extensions.yml"
      extension_function @function at @uri["func1"]
      // ...
    }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name, // corresponds to `anchor`
    StrAttr:$uri
  );
  let assemblyFormat = "$sym_name `at` $uri attr-dict";
}

class Substrait_ExtensionOp<string mnemonic, list<Trait> traits = []> :
  Substrait_Op<"extension_" # mnemonic, traits # [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<Symbol>
  ]> {
  let description = [{
      This op represents the `SimpleExtensionDeclaration` message type of
      Substrait along with the `Extension}]
      # snakeCaseToCamelCase<mnemonic>.ret #
    [{` message type in the `mapping_type` case. It is both a `Symbol` op, so it
      can be looked up in the symbol table of the plan it is contained in.
      Conversely, its symbol reference `uri` must refer to an extension URI op
      in the nearest symbol table.
    }];
  let arguments = (ins
    SymbolNameAttr:$sym_name, // corresponds to `anchor`
    FlatSymbolRefAttr:$uri,
    StrAttr:$name
  );
  let assemblyFormat = "$sym_name `at` $uri `[` $name `]` attr-dict";
  let extraClassDefinition = [{
    /// Implement `SymbolOpInterface`.
    ::mlir::LogicalResult $cppClass::verifySymbolUses(
        mlir::SymbolTableCollection &symbolTables) {
      if (!symbolTables.lookupNearestSymbolFrom<ExtensionUriOp>(*this,
                                                                getUriAttr()))
        return emitOpError() << "refers to " << getUriAttr()
                            << ", which is not a valid 'uri' op";
      return success();
    }
  }];
}

def Substrait_ExtensionFunctionOp : Substrait_ExtensionOp<"function"> {
  let summary = "Declares a simple extension function";
}

def Substrait_ExtensionTypeOp : Substrait_ExtensionOp<"type"> {
  let summary = "Declares a simple extension type";
}

def Substrait_ExtensionTypeVariationOp :
    Substrait_ExtensionOp<"type_variation"> {
  let summary = "Declares a simple extension type variation";
}

//===----------------------------------------------------------------------===//
// Plan
//===----------------------------------------------------------------------===//
// The definitions in this section are related to the top-level `Plan` message.
// See https://substrait.io/serialization/binary_serialization/ and
// https://github.com/substrait-io/substrait/blob/main/proto/substrait/plan.proto.
//===----------------------------------------------------------------------===//

def PlanBodyOp : AnyOf<[
    IsOp<"::mlir::substrait::PlanRelOp">,
    IsOp<"::mlir::substrait::ExtensionUriOp">,
    IsOp<"::mlir::substrait::ExtensionFunctionOp">,
    IsOp<"::mlir::substrait::ExtensionTypeOp">,
    IsOp<"::mlir::substrait::ExtensionTypeVariationOp">,
  ]>;

def Substrait_PlanOp : Substrait_Op<"plan", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    NoTerminator, NoRegionArguments, SingleBlock, SymbolTable
  ]> {
  let summary = "Represents a Substrait plan";
  let description = [{
    This op represents the `Plan` message type of Substrait. It carries the
    version information inline as attributes, so it also subsumes the `Version`
    message type. The body of the op consists of the `relation`s and the
    function and type extensions defined in the plan.
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
    the `Plan` message. The same op can represent either the `Rel`, in which
    case the `fieldNames` attribute is not set, or the `RootRel` case, in which
    case the `fieldNames` attribute corresponds to the `RelRoot.names` field.
    The body of this op contains various `RelOpInterface` ops (corresponding to
    the `Rel` message type) producing SSA values and the one being yielded
    reprents the root of the query tree that this op contains.
  }];
  let arguments = (ins OptionalAttr<StringArrayAttr>:$fieldNames);
  let regions = (region RegionOf<RelationBodyOp>:$body);
  let assemblyFormat = "(`as` $fieldNames^)? attr-dict-with-keyword $body";
  let hasRegionVerifier = 1;
  let builders = [
      OpBuilder<(ins ), [{
        build($_builder, $_state, ArrayAttr());
      }]>
    ];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    ::llvm::StringRef $cppClass::getDefaultDialect() {
      return SubstraitDialect::getDialectNamespace();
    }
  }];
}

def Substrait_YieldOp : Substrait_Op<"yield", [
    Terminator,
    ParentOneOf<[
      "::mlir::substrait::FilterOp",
      "::mlir::substrait::PlanRelOp",
      "::mlir::substrait::ProjectOp"
    ]>
  ]> {
  let summary = "Yields the result of a `PlanRelOp`";
  let arguments = (ins Variadic<AnyType>:$value);
  let assemblyFormat = "attr-dict ($value^ `:` type($value))?";
  let builders = [OpBuilder<(ins), [{ /* do nothing */ }]>];
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//
// The definitions in this section are related to the various `Expression`
// message types. See https://substrait.io/expressions/field_references/ and
// https://github.com/substrait-io/substrait/blob/main/proto/substrait/algebra.proto.
//===----------------------------------------------------------------------===//

// TODO(ingomueller): Make this a proper base for expressions.
class Substrait_ExpressionOp<string mnemonic, list<Trait> traits = []> :
  Substrait_Op<mnemonic, traits # [
    Substrait_ExpressionOpInterface, Pure
  ]>;

def Substrait_FieldReferenceOp : Substrait_ExpressionOp<"field_reference", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Field reference expression";
  let description = [{
    Represents a `FieldReference` message together with all messages it contains
    and the `Expression` message it is contained in.

    Example:

    ```mlir
    %0 = ...
    %1 = field_reference %0[[0]] : tuple<si32>
    // %1 is of type `si32`
    ```
  }];
  // TODO(ingomueller): extend to other types, map access, and masked references.
  let arguments = (ins
    Substrait_ContainerType:$container,
    DenseI64ArrayAttr:$position
  );
  let results = (outs Substrait_FieldType:$result);
  let assemblyFormat = [{
    $container `` $position attr-dict `:` type($container)
  }];
}

def Substrait_LiteralOp : Substrait_ExpressionOp<"literal", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Literal expression";
  let description = [{
    Represents a `Literal` message together with all messages it contains and
    the `Expression` message it is contained in.

    Example:

    ```mlir
    %0 = literal -1 : si1
    ```
  }];
  // TODO(ingomueller): extend to other types.
  let arguments = (ins Substrait_AtomicAttribute:$value);
  let results = (outs Substrait_AtomicType:$result);
  let assemblyFormat = "$value attr-dict";
}

def Substrait_CallOp : Substrait_ExpressionOp<"call", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  ]> {
  let summary = "Function call expression";
  let description = [{
    Represents a `ScalarFunction` message (or, in the future, other `*Function`
    messages) together with all messages it contains and the `Expression`
    message it is contained in.

    Currently, the specification of the function, which is in an external YAML
    file, is not taken into account, for example, to verify whether a matching
    overload exists or to verify/compute the result type.

    Example:

    ```mlir
    extension_uri @extension at "http://some.url/with/extensions.yml"
    extension_function @function at @extension["somefunc"]
    relation {
      // ...
      %1 = call @function(%0) : (tuple<si32>) -> si1
      // ...
    }
    ```
  }];
  // TODO(ingomueller): Add `FunctionOptions`.
  // TODO(ingomueller): Add support for `enum` and `type` argument types.
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<Substrait_FieldType>:$args
  );
  let results = (outs Substrait_FieldType:$result);
  let assemblyFormat = [{
    $callee `(` $args `)` attr-dict `:` `(` type($args) `)` `->` type($result)
  }];
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
    Substrait_RelOpInterface, Pure,
    PredOpTrait<"result must be extactly one Relation",
      And<[
        CPred<"$_op.getNumResults() == 1">,
        SubstLeaves<"$_self", "$_op.getResult(0).getType()",
          Substrait_Relation.predicate>
      ]>>
  ]>;

def Substrait_CrossOp : Substrait_RelOp<"cross", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Cross product operation";
  let description = [{
    Represents a `CrossRel` message together with the `RelCommon` and left and
    right `Rel` messages it contains.

    Example:

    ```mlir
    %0 = ...
    %1 = ...
    %2 = cross %0 x %1 : tuple<si32> x tuple<si32> -> tuple<si32, si32>
    ```
  }];
  let arguments = (ins
    Substrait_Relation:$left,
    Substrait_Relation:$right
  );
  let results = (outs Substrait_Relation:$result);
  let assemblyFormat = [{
    $left `x` $right attr-dict `:` type($left) `x` type($right)
  }];
}

def Substrait_EmitOp : Substrait_RelOp<"emit", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "Projection (a.k.a. 'emit') as dedicated operation";
  let description = [{
    Represents the `Emit` message of the `emit_kind` field in the `RelCommon`
    message. While projection is inlined into all relations in the protobuf
    format, this op separates out this functionality in a dedicated op in order
    to simplify rewriting.

    Example:

    ```mlir
    %0 = ...
    %1 = emit [2, 1] from %0 : tuple<si32, s1, si32> -> tuple<si32, si1>
    ```
  }];
  let arguments = (ins
    Substrait_Relation:$input,
    I64ArrayAttr:$mapping
  );
  let results = (outs Substrait_Relation:$result);
  let assemblyFormat = [{
    $mapping `from` $input attr-dict `:` type($input) `->` type($result)
  }];
  let hasFolder = 1;
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    ::llvm::StringRef $cppClass::getDefaultDialect() {
      return SubstraitDialect::getDialectNamespace();
    }
  }];
}

def Substrait_FilterOp : Substrait_RelOp<"filter", [
    SingleBlockImplicitTerminator<"::mlir::substrait::YieldOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>,
    SameOperandsAndResultType
  ]> {
  let summary = "Filter operation";
  let description = [{
    Represents a `FilterRel` message together with the `RelCommon`, input `Rel`,
    and `Expression` messages it contains.

    Example:

    ```mlir
    %0 = ...
    %1 = filter %0 : tuple<si32> {
    ^bb0(%arg : tuple<si32>):
      %true = literal -1 : si1
      yield %true : si1
    }
    ```
  }];
  let arguments = (ins Substrait_Relation:$input);
  let regions = (region AnyRegion:$condition);
  let results = (outs Substrait_Relation:$result);
  // TODO(ingomueller): We could elide/shorten the block argument from the
  //                    assembly by writing custom printers/parsers similar to
  //                    `scf.for` etc.
  let assemblyFormat = [{
    $input attr-dict `:` type($input) $condition
  }];
  let hasRegionVerifier = 1;
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    ::llvm::StringRef $cppClass::getDefaultDialect() {
      return SubstraitDialect::getDialectNamespace();
    }
  }];
}

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

def Substrait_ProjectOp : Substrait_RelOp<"project", [
    SingleBlockImplicitTerminator<"::mlir::substrait::YieldOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getDefaultDialect"]>
  ]> {
  let summary = "Project operation";
  let description = [{
    Represents a `ProjectRel` message together with the `RelCommon`, input
    `Rel`, and `Expression` messages it contains. While in protobuf the
    different `Expression` messages are distinct trees, the `project` op has
    a single `expression` region with one terminating `yield` and the values
    yielded by that terminator correspond to the expressions. Each individual
    `Expression` thus corresponds to the whole use-def tree of the corresponding
    yielded value.

    Example:

    ```mlir
    %0 = ...
    %1 = project %0 : tuple<si32> -> tuple<si32, si1, si32> {
    ^bb0(%arg : tuple<si32>):
      %true = literal -1 : si1
      %42 = literal 42 : si32
      yield %true, %42 : si1, si32
    }
    ```
  }];
  let arguments = (ins Substrait_Relation:$input);
  let regions = (region AnyRegion:$expressions);
  let results = (outs Substrait_Relation:$result);
  // TODO(ingomueller): We could elide/shorten the block argument from the
  //                    assembly by writing custom printers/parsers similar to
  //                    `scf.for` etc.
  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($result) $expressions
  }];
  let hasRegionVerifier = 1;
  let hasFolder = 1;
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    ::llvm::StringRef $cppClass::getDefaultDialect() {
      return SubstraitDialect::getDialectNamespace();
    }
  }];
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITOPS
