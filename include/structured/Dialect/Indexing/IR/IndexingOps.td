//===-- IndexingOps.td - Indexing operations definitions ---*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_INDEXING_IR_INDEXINGOPS
#define STRUCTURED_DIALECT_INDEXING_IR_INDEXINGOPS

include "structured/Dialect/Indexing/IR/IndexingDialect.td"
include "structured/Dialect/Indexing/IR/IndexingTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Dialect/Tensor/IR/TensorBase.td"

class Indexing_Op<string mnemonic, list<Trait> traits = []> :
        Op<Indexing_Dialect, mnemonic, traits>;

def Indexing_GatherOp : Indexing_Op<"gather", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "gather a subset of a tensor at specified indices";
  let description = [{ }];

  let arguments = (ins AnyRankedTensor:$source,
                       RankedTensorOf<[AnySignlessIntegerOrIndex]>:$indices,
                       DenseI64ArrayAttr:$gather_dims,
                       UnitAttr:$unique);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $indices `]` `gather_dims` `(` $gather_dims `)` (`unique` $unique^)? attr-dict
        `:` functional-type(operands, results)
  }];
}

def Indexing_ConcatenateOp : Indexing_Op<"concatenate",
    [Pure, SameOperandsAndResultElementType,
     DeclareOpInterfaceMethods<InferTypeOpInterface>]> {
  let summary = "Concatenate operation";
  let description = [{
    Concatenates a variadic number of tensors in `inputs` along `dimension`
    dimension in the same order as the given arguments and produces a `result`
    tensor.

    Example:
    ```mlir
    %result = indexing.concatenate %input0, %input1, dim = 0
        : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    ```
  }];

  let arguments = (ins
    Variadic<AnyRankedTensor>:$inputs,
    I64Attr:$dimension
  );

  let results = (outs AnyRankedTensor);

  let assemblyFormat = [{
     `(` $inputs `)` `{` `dim` `=` $dimension `}` attr-dict `:` functional-type(operands, results)
  }];
}

#endif // STRUCTURED_DIALECT_INDEXING_IR_INDEXINGOPS
