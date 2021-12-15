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

def Linalg_Transform_Dialect : Dialect {
  let name = "linalg_transform";
  let cppNamespace = "::mlir::linalg::transform";
  let dependentDialects = [
    "linalg::LinalgDialect",
  ];
}

class Linalg_Transform_Operation<string name, list<OpTrait> props = []>
    : Op<Linalg_Transform_Dialect, name, props> {
  let cppNamespace = "::mlir::linalg::transform";
}

//===----------------------------------------------------------------------===//

def SequenceOp : Linalg_Transform_Operation<"sequence",
                                            [NoTerminator, OpAsmOpInterface]> {
  let description = [{Contains a sequence of transformation ops to apply.}];

  let regions = (region SizedRegion<1>:$body);
  let assemblyFormat = "attr-dict-with-keyword regions";

  let extraClassDeclaration = [{
    static StringRef getDefaultDialect() { return "linalg_transform"; }
  }];
}

def ApplyOp : Linalg_Transform_Operation<"apply", [NoTerminator]> {
  let description = [{A container operation that connects a sequence of
  transformations to apply to the condition at which the operation should
  be applied. The condition should be a PDLMatchOp producing the root
  operation, which is forwarded as the unique basic block argument to the
  `transforms` region of the operation.}];

  let regions = (region SizedRegion<1>:$transforms, SizedRegion<1>:$condition);
  let assemblyFormat = "$transforms `when` $condition attr-dict-with-keyword";
}

def PDLMatchOp : Linalg_Transform_Operation<"pdl_match"> {
  let description = [{Specifies the name of the PDL pattern used to match the
  root operation on which the transformations are applied.}];

  let arguments = (ins SymbolRefAttr:$pattern);
  let assemblyFormat = "$pattern attr-dict";
}

//===----------------------------------------------------------------------===//

def TileOp : Linalg_Transform_Operation<"tile"> {
  let description = [{Indicates that ops of a specific kind in the given
  function should be tiled with the options provided as attributes.}];

  let arguments = (ins PDL_Operation:$op,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$sizes,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$interchange,
                   DefaultValuedAttr<BoolAttr, "false">:$pad,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$pack_paddings,
                   DefaultValuedAttr<I64ArrayAttr, "{}">:$hoist_paddings,
                   DefaultValuedAttr<BoolAttr, "false">:$scalarize_dyn_dims,
                   DefaultValuedAttr<BoolAttr, "false">:$generalize
                  );
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$op attr-dict";
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

def VectorizeOp : Linalg_Transform_Operation<"vectorize"> {
  let description = [{Indiactes that ops of a specific kind in the given
  function should be vectorized with the options provided as attributes.}];

  let arguments = (ins PDL_Operation:$op,
                   DefaultValuedAttr<BoolAttr, "false">:$vectorize_padding
                  );
  let results = (outs PDL_Operation:$transformed);

  let assemblyFormat = "$op attr-dict";
}

def LowerVectorsOp : Linalg_Transform_Operation<"lower_vectors"> {
  let description = [{Indicates that the vector operations in the entire
  module should be lowered to simpler primitives (multiple stages of lowering
  be executed at once).}];

  let arguments =
    (ins DefaultValuedAttr<I64ArrayAttr, "{0, 1, 2, 3, 4, 5, 6}">:$stages,
     DefaultValuedAttr<StrAttr, "\"outerproduct\"">:$contraction_lowering,
     DefaultValuedAttr<StrAttr, "\"innerparallel\"">:$multireduction_lowering,
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

#endif // LINALG_TRANSFORM_OPS
