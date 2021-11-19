//===-- VectorExtOps.td - Vector Extension dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_VECTOREXT_VECTOREXTOPS
#define RUNNERS_VECTOREXT_VECTOREXTOPS

include "VectorExt/VectorExtBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VectorExt_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<VectorExt_Dialect, mnemonic, traits> {
}

def VectorExt_YieldOp : VectorExt_Op<"yield", [
    NoSideEffect, ReturnLike, Terminator]> {
  let summary = "VectorExt yield op";
  let description = [{
    `vector_ext.yield` is a special terminator operation for blocks inside
    regions in `vector_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // RUNNERS_VECTOREXT_VECTOREXTOPS
