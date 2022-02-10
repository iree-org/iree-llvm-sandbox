//===- LinalgTransform.td - Linalg transform dialect -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Definitions of attributes, operations and types used to control
// transformations on the Linalg dialect.
//
//===----------------------------------------------------------------------===//

#ifndef LINALG_TRANSFORM_OPS
#define LINALG_TRANSFORM_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Dialect/PDL/IR/PDLTypes.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def Linalg_Transform_Dialect : Dialect {
  let name = "linalg_transform";
  let cppNamespace = "::mlir::linalg::transform";
  let dependentDialects = [
    "linalg::LinalgDialect",
  ];
}

// Operations with this trait must provide the following methods:
//   - `Value target()` - returns the operation handle (value of !pdl.operation
//     type) targeted by this transformation, if available;
//   - `Optional<SymbolRefAttr> matcher()` - returns the name of the PDL matcher
//     that selects the ops targeted by this transformation, if provided.
class Linalg_Transform_Operation<string name, list<Trait> props = []>
    : Op<Linalg_Transform_Dialect, name, props> {
  let cppNamespace = "::mlir::linalg::transform";
}

//===----------------------------------------------------------------------===//

def ScopeOp : Linalg_Transform_Operation<"util.scope",
    [IsolatedFromAbove, DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let description = [{An operation to restrict transformation scopes.}];

  let regions = (region AnyRegion:$body);
  let arguments = (ins Variadic<AnyType>:$ins);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = "`(` operands `)` attr-dict-with-keyword $body `:` functional-type(operands, results)";

  let verifier = [{ return RegionBranchOpInterface::verifyTypes(*this); }];
}

def ForwardOp : Linalg_Transform_Operation<"util.forward",
                                           [Terminator, HasParent<"ScopeOp">]> {
  let description = [{Terminator for a scope operation, indicating the results
                      that should be forwarded out of the scope.}];

  let arguments = (ins Variadic<AnyType>:$ins);
  let assemblyFormat = "operands attr-dict `:` type(operands)";
}

//===----------------------------------------------------------------------===//

def SequenceOp : Linalg_Transform_Operation<"sequence",
                                            [NoTerminator, OpAsmOpInterface]> {
  let description = [{Contains a sequence of transformation ops to apply.

  The transformations indicated by the sequence are applied in order of their
  appearance. Each value produced by a transformation within the sequence
  corresponds to an operation or a group of operations in the IR being
  transformed. Therefore, each value may be used at most once by another
  transformation operation as the transformation is likely to replace the
  transformed operation with another operation or a group thereof. In such
  cases, the tranfsormation operation is expected to produce a new value to
  denote the newly produced operations that can be transformed further.
  }];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword regions";

  let extraClassDeclaration = [{
    static StringRef getDefaultDialect() { return "linalg_transform"; }
  }];

  let verifier = [{ return verifySequenceOp(*this); }];
}

//===----------------------------------------------------------------------===//

def TargetableOpTrait : NativeOpTrait<"TargetableTransformOpTrait"> {
  let cppNamespace = "::mlir";
}

def TileOp : Linalg_Transform_Operation<"tile", [TargetableOpTrait]> {
  let description = [{Indicates that ops of a specific kind in the given
  function should be tiled with the options provided as attributes.}];

  let arguments = (ins Optional<PDL_Operation>:$target,
                   OptionalAttr<SymbolRefAttr>:$targetMatcher,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$peel,
                   DefaultValuedAttr<BoolAttr, "false">:$pad,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$pack_paddings,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$hoist_paddings,
                   DefaultValuedAttr<
                      TypedArrayAttrBase<I64ArrayAttr,
                                         "array of arrays of i64">,
                      "{}">:$transpose_paddings,
                   DefaultValuedAttr<BoolAttr, "false">:$scalarize_dyn_dims,
                   DefaultValuedAttr<BoolAttr, "false">:$generalize
                  );
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "($target^)? (`when` $targetMatcher^)? attr-dict";

  let hasVerifier = 1;
}

def BufferizeOp : Linalg_Transform_Operation<"bufferize"> {
  let description = [{Indicates that the entire module should be bufferized.}];
  let assemblyFormat = "attr-dict";
}

def DecomposeOp : Linalg_Transform_Operation<"decompose"> {
  let description = [{Indicates that ops in the entire module should be
  decomposed into lower-level components.}];
  let assemblyFormat = "attr-dict";
}

def VectorizeOp : Linalg_Transform_Operation<"vectorize", [TargetableOpTrait]> {
  let description = [{Indiactes that ops of a specific kind in the given
  function should be vectorized with the options provided as attributes.}];

  let arguments = (ins Optional<PDL_Operation>:$target,
                   OptionalAttr<SymbolRefAttr>:$targetMatcher,
                   DefaultValuedAttr<BoolAttr, "false">:$vectorize_padding
                  );
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "($target^)? (`when` $targetMatcher^)? attr-dict";
}

def LowerVectorsOp : Linalg_Transform_Operation<"lower_vectors"> {
  let description = [{Indicates that the vector operations in the entire
  module should be lowered to simpler primitives (multiple stages of lowering
  be executed at once).}];

  let arguments =
    (ins DefaultValuedAttr<I64ArrayAttr, "{0, 1, 2, 3, 4, 5, 6}">:$stages,
     DefaultValuedAttr<StrAttr, "\"outerproduct\"">:$contraction_lowering,
     DefaultValuedAttr<StrAttr, "\"innerparallel\"">:$multireduction_lowering,
     DefaultValuedAttr<StrAttr, "\"linalg-copy\"">:$split_transfers,
     DefaultValuedAttr<BoolAttr, "true">:$unroll_vector_transfers,
     DefaultValuedAttr<StrAttr, "\"eltwise\"">:$transpose_lowering,
     DefaultValuedAttr<BoolAttr, "false">:$transpose_avx2_lowering
    );

  let assemblyFormat = "attr-dict";
}

def LowerToLLVMOp : Linalg_Transform_Operation<"lower_to_llvm"> {
  let description = [{Indicates that the entire module should be converted
  to the LLVM dialect. This is expected to be the last transformation in
  a sequence.}];

  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//

def ExpertOp : Linalg_Transform_Operation<"expert", [TargetableOpTrait]> {
  let description = [{A "transformation expert" that can be lowered to a
  sequence of transformations. The details of the lowering depend on the name
  and are expressed declaratively.}];

  let arguments = (ins Optional<PDL_Operation>:$target,
                   OptionalAttr<SymbolRefAttr>:$targetMatcher,
                   StrAttr:$expertName);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "`apply` $expertName (`to` $target^)? "
                       "(`when` $targetMatcher^)? attr-dict";
}

#endif // LINALG_TRANSFORM_OPS
