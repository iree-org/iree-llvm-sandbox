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
  }];
  let constructor = "mlir::createDecomposeTuplesPass()";
  let dependentDialects = [
  ];
}

#endif // TUPLE_TRANSFORMS_PASSES
