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

def VectorExt_WarpSingleLaneOp : VectorExt_Op<"warp_execute_on_lane_0",
      [DeclareOpInterfaceMethods<RegionBranchOpInterface, ["areTypesCompatible"]>,
       SingleBlockImplicitTerminator<"vector_ext::YieldOp">,
       RecursiveSideEffects]> {
  let summary = "vector.warp_execute_on_lane_0 operation";
  let description = [{
    Operation that encapsulates code that should be run on a single lane within
    a warp. Any code present in the region would only be executed on first lane
    based on the `laneid` operand. The `laneid` operand is an integer ID between
    [0, `warp_size`). The `warp_size` attribute indicates the number of lanes in
    a warp.

    // TODO: Support multiple dimensions, i.e., one lane ID per dimension.

    Operands are vector values distributed on all threads that may be used by
    the single lane execution. The matching region argument is a vector of all
    the values of those threads available to the single active lane. The
    distributed dimension is implicit based on the shape of the operand and
    argument. In the future this may be described by an affine map.

    Return values are distributed on all threads using laneId as index. The
    vector is distributed based on the shape ratio between the vector type of
    the yield and the result type.
    If the shapes are the same this means the value is broadcasted to all lanes.
    In the future the distribution can be made more explicit using affine_maps
    and will support having multiple Ids.

    During lowering values passed as operands and return value need to be
    visible to different lanes within the warp. This would usually be done by
    going through memory.

    The region is *not* isolated from above. For values coming from the parent
    region only the lane 0 value will be available so it generally only make
    sense for uniform values. If all the lanes need to be accessible the value
    needs to go through region argument.

    Example:
    ```
    vector_ext.warp_execute_on_lane_0 (%laneid)[32] {
      ...
    }
    ```

    This may be lowered to an scf.if region as below:
    ```
      %cnd = arith.cmpi eq, %laneid, %c0 : index
      scf.if %cnd {
         ...
      }
    ```

    When the region has operands and/or return values:
    ```
    %0 = vector_ext.warp_execute_on_lane_0(%laneid)[32]
    args(%v0 : vector<4xi32>) -> (vector<1xf32>) {
    ^bb0(%arg0 : vector<128xi32>) :
      ...
      vector_ext.yield %1 : vector<32xf32>
    }
    ```

    values at the region boundary would go through memory:
    ```
    %tmp0 = memreg.alloc() : memref<32xf32, 3>
    %tmp1 = memreg.alloc() : memref<32xf32, 3>
    %cnd = arith.cmpi eq, %threadid, %c0 : index
    vector.store %v0, %tmp0[%threadid] : memref<32xf32>, vector<1xf32>
    warp_sync
    scf.if %cnd {
      %arg0 = vector.load %tmp0[%c0] : memref<32xf32>, vector<32xf32>
      ...
      vector.store %1, %tmp1[%c0] : memref<32xf32>, vector<32xf32>
    }
    warp_sync
    %0 = vector.load %tmp1[%threadid] : memref<32xf32>, vector<32xf32>
    ```

  }];

  let hasVerifier = 1;
  let arguments = (ins Index:$laneid, I64Attr:$warp_size,
                       Variadic<AnyType>:$args);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$warpRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "Value":$laneid, "int64_t":$warpSize)>,
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$laneid,
                   "int64_t":$warpSize)>,
    // `blockArgTypes` are different than `args` types as they are they
    // represent all the `args` instances visibile to lane 0. Therefore we need
    // to explicit pass the type.
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$laneid,
                   "int64_t":$warpSize, "ValueRange":$args,
                   "TypeRange":$blockArgTypes)>
  ];

  let extraClassDeclaration = [{
    bool isDefinedOutsideOfRegion(Value value) {
      return !getRegion().isAncestor(value.getParentRegion());
    }
  }];

}

#endif // DIALECTS_VECTOREXT_VECTOREXTOPS
