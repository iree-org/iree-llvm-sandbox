//===-- VectorExtOps.td - Vector Extension dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_VECTOREXT_VECTOREXTOPS
#define DIALECTS_VECTOREXT_VECTOREXTOPS

include "Dialect/VectorExt/VectorExtBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class VectorExt_Op<string mnemonic, list<Trait> traits = []> :
    Op<VectorExt_Dialect, mnemonic, traits> {
  let hasCustomAssemblyFormat = 1;
}

def VectorExt_PredicateOp : VectorExt_Op<"predicate",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface>,
       SingleBlockImplicitTerminator<"vector_ext::YieldOp">, RecursiveSideEffects]> {
  let summary = "vector.predicate operation";
  let description = [{ TODO }];

  let arguments = (ins VectorOf<[I1]>:$predicateMask, Variadic<Index>:$indices,
                       VectorOf<[I1]>:$incomingMask);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$truePredicateRegion);

  // TODO: Add falsePredicateRegion.

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$predicateMask, "ValueRange":$indices,
                   "Value":$incomingMask)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$predicateMask,
                   "ValueRange":$indices, "Value":$incomingMask,
                   CArg<"function_ref<void(OpBuilder &, Location)>",
                        "buildTerminatedBody">:$truePredicateBuilder)>
  ];
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

#endif // DIALECTS_VECTOREXT_VECTOREXTOPS
