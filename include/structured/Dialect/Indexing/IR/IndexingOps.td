//===-- IndexingOps.td - Indexing operations definitions -*- tablegen -*-===//
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

//===----------------------------------------------------------------------===//
// Debugging/testing utilities
//===----------------------------------------------------------------------===//

def Indexing_GatherOp : Indexing_Op<"gather", [
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>
  ]> {
  let summary = "gather a subset of a tensor at specified indices";
  let description = [{ }];

  let arguments = (ins AnyRankedTensor:$source,
                       RankedTensorOf<[AnySignlessIntegerOrIndex]>:$indices,
                       DenseI64ArrayAttr:$coordinates);
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $indices `]` `coordinates` `=` $coordinates attr-dict `:` functional-type(operands, results)
  }];
}


#endif // STRUCTURED_DIALECT_INDEXING_IR_INDEXINGOPS
