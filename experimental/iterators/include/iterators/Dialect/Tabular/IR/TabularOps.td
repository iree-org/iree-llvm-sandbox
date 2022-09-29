//===-- TabularOps.td - Tabular operations definitions -----*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TABULAR_DIALECT_TABULAR_IR_TABULAROPS
#define TABULAR_DIALECT_TABULAR_IR_TABULAROPS

include "iterators/Dialect/Tabular/IR/TabularDialect.td"
include "iterators/Dialect/Tabular/IR/TabularTypes.td"
include "mlir/IR/OpBase.td"

class Tabular_Op<string mnemonic, list<Trait> traits = []> :
  Op<Tabular_Dialect, mnemonic, traits> {
}

def Tabular_CreateTabularViewOp
    : Tabular_Op<"create_tabular_view"> {
  let summary = "Creates a `tabular_view` from the given memrefs";
  let description = [{
    Converts a variadic number of memrefs of rank 1 into the columns of a
    `tabular_view` with corresponding column types. This currently only works
    for memrefs that can be represented with a bare pointer, i.e., those with a
    contiguous memory layout. Furthermore, no dynamic dimension is allowed
    currently (in order to avoid runtime checks).

    Example:
    ```mlir
      %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
      %t2 = arith.constant dense<[3, 4, 5]> : tensor<3xi64>
      %m1 = bufferization.to_memref %t1 : memref<3xi32>
      %m2 = bufferization.to_memref %t2 : memref<3xi64>
      %view = "tabular.create_tabular_view"(%m1, %m2)
        : (memref<3xi32>, memref<3xi64>)
          -> !tabular.tabular_view<i32,i64>
    ```
  }];
  let arguments = (ins Variadic<AnyMemRef>:$memrefs);
  let results = (outs Tabular_TabularView:$view);
  let hasVerifier = true;
  let assemblyFormat = [{
    operands attr-dict `:` functional-type($memrefs, $view)
  }];
}

#endif // TABULAR_DIALECT_TABULAR_IR_TABULAROPS
