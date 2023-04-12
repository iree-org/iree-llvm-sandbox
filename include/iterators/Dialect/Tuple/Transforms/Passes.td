//===-- Passes.td - Transform pass definition file ---------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TUPLE_TRANSFORMS_PASSES
#define TUPLE_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DecomposeTuples
//===----------------------------------------------------------------------===//

def DecomposeTuples : Pass<"decompose-tuples", "ModuleOp"> {
  let summary = "Decompose tuples into their constituent elements";
  let description = [{
    Decomposes potentially nested tuples into the elements they are made of,
    i.e., transforms each tuple into a set of SSA values where each value
    corresponds to exactly one (non-tuple) element of the input tuple. The types
    of the resulting SSA values correspond exactly to what
    `tupleType.getFlattenedTypes()` returns. Depending on the pass options, this
    decomposition is carried through func and SCF ops.

    Example:

    ```mlir
    !nested_tuple = tuple<tuple<>, i32, tuple<i64>>
    func.func @extractNested(%input: !nested_tuple) -> tuple<i64> {
      %element = tuple.extract %input[2] : !nested_tuple
      return %element : tuple<i64>
    }
    ```

    is transformed to

    ```mlir
    func.func @extractNested(%arg0: i32, %arg1: i64) -> i64 {
      return %arg1 : i64
    }
    ```
  }];
  let options = [
    Option<"convertFuncOps", "convert-func-ops", "bool", /*default=*/"true",
           "Decompose tuples across function boundaries.">,
    Option<"convertSCFOps", "convert-scf-ops", "bool", /*default=*/"true",
           "Decompose tuples across SCF ops.">,
  ];
  let constructor = "mlir::createDecomposeTuplesPass()";
  let dependentDialects = [];
}

#endif // TUPLE_TRANSFORMS_PASSES
