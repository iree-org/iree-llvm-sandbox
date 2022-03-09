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

include "Dialects/LinalgTransform/TransformOpInterface.td"
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

class Transform_Op<string name, list<Trait> props = []>
    : Linalg_Transform_Operation<name, !listconcat(props, [
      DeclareOpInterfaceMethods<TransformOpInterface, ["apply"]>])>;

//===----------------------------------------------------------------------===//

def ScopeOp : Linalg_Transform_Operation<"util.scope",
    [IsolatedFromAbove, DeclareOpInterfaceMethods<RegionBranchOpInterface>]> {
  let description = [{An operation to restrict transformation scopes.}];

  let regions = (region AnyRegion:$body);
  let arguments = (ins Variadic<AnyType>:$ins);
  let results = (outs Variadic<AnyType>:$outs);
  let assemblyFormat = [{ `(` operands `)` attr-dict-with-keyword $body
                          `:` functional-type(operands, results) }];
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

def MatchOp : Transform_Op<"match"> {
  let description = [{ Find and return an op that matches the specific PDL
  pattern. When executed inside a sequence, it returns all matching ops. }];

  let arguments = (ins SymbolRefAttr:$targetMatcher);
  let results = (outs PDL_Operation:$target);

  let assemblyFormat = "$targetMatcher attr-dict";
}

//===----------------------------------------------------------------------===//

def TileOp : Linalg_Transform_Operation<"tile",
    [TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Indicates that ops of a specific kind in the given
  function should be tiled with the options provided as attributes.}];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$peel,
                   DefaultValuedAttr<BoolAttr, "false">:$scalarize_dyn_dims);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def FuseOp : Linalg_Transform_Operation<"fuse",
    [TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Tiles the operations pointed to by the target handle and
  fuses their producers greedily using the options provided as attributes.}];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$tile_interchange
                  );
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def GeneralizeOp : Linalg_Transform_Operation<"generalize",
    [TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Generalizes the operations pointed to
  by the target handle.}];

  let arguments = (ins PDL_Operation:$target);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def InterchangeOp : Linalg_Transform_Operation<"interchange",
    [TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Interchanges the iterators of the operations pointed to
  by the target handle using the iterator interchange attribute.}];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$iterator_interchange);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def PadOp : Linalg_Transform_Operation<"pad",
    [TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Pads the operations pointed to by the target handle
  using the options provides as operation attributes.}];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$pack_paddings,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$hoist_paddings,
                   DefaultValuedAttr<
                      TypedArrayAttrBase<I64ArrayAttr,
                                         "array of arrays of i64">,
                      "{}">:$transpose_paddings);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::linalg::LinalgOp> applyToOne(
        ::mlir::linalg::LinalgOp target);
  }];
}

def BufferizeOp : Transform_Op<"bufferize"> {
  let description = [{Indicates that the entire module should be bufferized.}];
  let assemblyFormat = "attr-dict";
}

def DecomposeOp : Transform_Op<"decompose"> {
  let description = [{Indicates that ops in the entire module should be
  decomposed into lower-level components.}];
  let assemblyFormat = "attr-dict";
}

def VectorizeOp : Transform_Op<"vectorize"> {
  let description = [{Indiactes that vectorization should be performed. If a
  target handle is provided, only vectorizes the operations pointed to by the
  handle. Otherwise vectorizes the entire module. Vectorization options are
  provided as operation attributes.}];

  let arguments = (ins Optional<PDL_Operation>:$target,
                   DefaultValuedAttr<BoolAttr, "false">:$vectorize_padding
                  );
  let results = (outs Optional<PDL_Operation>:$transformed);

  let hasCustomAssemblyFormat = 1;
}

def LowerVectorsOp : Transform_Op<"lower_vectors"> {
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

def LowerToLLVMOp : Transform_Op<"lower_to_llvm"> {
  let description = [{Indicates that the entire module should be converted
  to the LLVM dialect. This is expected to be the last transformation in
  a sequence.}];
  
  let arguments =
    (ins DefaultValuedAttr<BoolAttr, "false">:$reassociate_fp_reductions,
     DefaultValuedAttr<BoolAttr, "false">:$enable_index_optimizations,
     DefaultValuedAttr<BoolAttr, "false">:$enable_arm_neon,
     DefaultValuedAttr<BoolAttr, "false">:$enable_arm_sve,
     DefaultValuedAttr<BoolAttr, "false">:$enable_amx,
     DefaultValuedAttr<BoolAttr, "false">:$enable_x86vector
    );

  let assemblyFormat = "attr-dict";
}

def GetParentLoopOp : Linalg_Transform_Operation<"get_parent_loop", [
    TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Obtains a handle to a parent loop of the given
  operation.}];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<Confined<I64Attr, [IntPositive]>,
                                     "1">:$num_loops);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(Operation *source);
  }];
}

def UnrollLoopOp : Linalg_Transform_Operation<"unroll_loop", [
    TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Unrolls the given loop with the given unroll factor.}];

  let arguments = (ins PDL_Operation:$target,
                   Confined<I64Attr, [IntPositive]>:$factor);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    LogicalResult applyToOne(::mlir::scf::ForOp loop);
  }];
}

def PipelineLoopOp : Linalg_Transform_Operation<"pipeline_loop", [
    TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64Attr, "1">:$iteration_interval,
                   DefaultValuedAttr<I64Attr, "10">:$read_latency);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::scf::ForOp> applyToOne(::mlir::scf::ForOp loop);
  }];
}

def OutlineLoopOp : Linalg_Transform_Operation<"outline_loop", [
    DeclareOpInterfaceMethods<TransformOpInterface, ["apply"]>]> {
  let arguments = (ins PDL_Operation:$target,
                   StrAttr:$func_name);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<::mlir::FuncOp> applyToOne(::mlir::scf::ForOp loop);
  }];
}

//===----------------------------------------------------------------------===//
// LinalgExt specific transforms
//===----------------------------------------------------------------------===//

def TileToLinalgExtTileOp : Linalg_Transform_Operation<"tile_to_linalg_ext_tile_op",
    [TransformOpInterface, TargetableSingleOperandTransformOpTrait]> {
  let description = [{Tile a linalg op with linalg_ext.tile op along a single 
  dimension.}];
  
  let summary = [{
    0 should be used as a tile size to skip tiling a particular dimension.
    This is a commonly used convention in Linalg.
    For the purpose of this transformation, a single non-zero positive tile 
    size is allowed.
    }];

  let arguments = (ins PDL_Operation:$target,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$sizes);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$target attr-dict";

  let extraClassDeclaration = [{
    ::mlir::FailureOr<Operation *> applyToOne(
        ::mlir::TilingInterface target);
  }];
}

//===----------------------------------------------------------------------===//

def ExpertOp : Linalg_Transform_Operation<"expert"> {
  let description = [{A "transformation expert" that can be lowered to a
  sequence of transformations. The details of the lowering depend on the name
  and are expressed declaratively.}];

  let arguments = (ins PDL_Operation:$target,
                   StrAttr:$expertName);
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "`apply` $expertName `to` $target attr-dict";
}

#endif // LINALG_TRANSFORM_OPS
