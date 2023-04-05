//===-- Passes.td - Transform pass definition file ---------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_TRANSFORMS_PASSES
#define ITERATORS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IteratorsToLLVM
//===----------------------------------------------------------------------===//

def DecomposeIteratorStates : Pass<"decompose-iterator-states", "ModuleOp"> {
  let summary = "Decompose iterator states into their constituent values";
  let description = [{
    Iterator state "bundle" values that constitute the current state of
    iterators, which often includes the state of nested iterators. This pass
    decomposes these bundles into their constituent values such that the
    `iterators.state` type is completely eliminated. In particular, the
    creation, field access, and field updates now simply forward SSA values,
    which are then carried as individual arguments through `scf` and `func` ops.
    This decomposition allows further passes to run without knowing anything
    about iterators, i.e., it makes iterators composable with other passes.

    Example:

    ```mlir
    func.func @example(%arg : !iterators.state<i1, i32>) -> (!iterators.state<i1, i32>) {
      %i1 = iterators.extractvalue %arg[0] : !iterators.state<i1, i32>
      %result = scf.if %i1 -> !iterators.state<i1, i32> {
        scf.yield %arg : !iterators.state<i1, i32>
      } else {
        %true = arith.constant 1 : i1
        %updated = iterators.insertvalue %true into %arg[0] : !iterators.state<i1, i32>
        scf.yield %updated : !iterators.state<i1, i32>
      }
      return %result : !iterators.state<i1, i32>
    }
    ```

    gets decomposed into

    ```mlir
    func.func @example(%arg0: i1, %arg1: i32) -> (i1, i32) {
      %true = arith.constant true
      %0:2 = scf.if %arg0 -> (i1, i32) {
        scf.yield %arg0, %arg1 : i1, i32
      } else {
        scf.yield %true, %arg1 : i1, i32
      }
      return %0#0, %0#1 : i1, i32
    }
    ```
  }];
  let constructor = "mlir::createDecomposeIteratorStatesPass()";
  let dependentDialects = [
    "scf::SCFDialect",
    "func::FuncDialect",
  ];
}

#endif // ITERATORS_TRANSFORMS_PASSES
