//===-- LinalgExtOps.td - Linalg Extension dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_LINALGEXT_LINALGEXTOPS
#define DIALECTS_LINALGEXT_LINALGEXTOPS

include "Dialects/LinalgExt/LinalgExtBase.td"
include "Dialects/LinalgExt/LinalgExtInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"

class LinalgExt_PureOp<string mnemonic, list<Trait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits> {
}

class LinalgExt_Op<string mnemonic, list<Trait> traits = []> :
    LinalgExt_PureOp<mnemonic,
       !listconcat(traits,
       [LinalgExtInterface, SingleBlockImplicitTerminator<"YieldOp">])> {
  let verifier = [{ return success(); }];
  code extraLinalgExtOpClassDeclaration = [{
    SmallVector<Value> getDestinationOperands(OpBuilder &b) {
      SmallVector<Value> dest(outputs().begin(), outputs().end());
      return dest;
    }
  }];
}

def LinalgExt_ReverseOp : LinalgExt_Op<"reverse", [
    AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<TilingInterface, [
        "getLoopIteratorTypes",
        "getIterationDomain",
        "getTiledImplementation",
        "getOutputTileSizes"]>]> {
  let summary = "Reverse operator";
  let description = [{
    Operation to reverse the values along specified dimension(s).

    Reverse operations cannot be represented as `LinalgOp` since the
    indexing maps are not handled by the tiling algorithm in
    Linalg. Defining this as a separate operation to use the
    `TilingInterface` to tile the operation.
  }];

  let arguments = (ins Variadic<AnyShaped>:$inputs,
                       Variadic<AnyShaped>:$outputs,
                       I64ElementsAttr:$dimensions
  );
  let results = (outs Variadic<AnyRankedTensor>:$results);
  let assemblyFormat = [{
    `dimensions` `(` $dimensions `)`
    attr-dict (`ins` `(` $inputs^ `:` type($inputs) `)`)?
    (`outs` `(` $outputs^ `:` type($outputs) `)`)?
    (`:` type($results)^)?
  }];
  let extraClassDeclaration = extraLinalgExtOpClassDeclaration # [{
    Value input() {
      return getInputOperand(0)->get();
    }
    Value output() {
      return getOutputOperand(0)->get();
    }
    ShapedType getOperandType() {
      return input().getType().cast<ShapedType>();
    }
    int64_t getOperandRank() {
      return getOperandType().getRank();
    }
    ArrayRef<int64_t> getOprerandShape() {
      return getOperandType().getShape();
    }
    SmallVector<int64_t> dims() {
      SmallVector<int64_t> ret;
      for (const APInt& elem : dimensions()) {
        ret.push_back(elem.getLimitedValue());
      }
      return ret;
    };
  }];
}

def LinalgExt_YieldOp : LinalgExt_PureOp<"yield", [
    NoSideEffect, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `linalg_ext.yield` is a special terminator operation for blocks inside
    regions in `linalg_ext` ops.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def LinalgExt_TileOp : LinalgExt_PureOp<"tile",
      [
       // TODO: enable to allow hoisting, LICM and isDefinedOutside
       // DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"linalg_ext::TileYieldOp">,
       RecursiveSideEffects
      ]> {
  let summary = "tile operation";
  let description = [{
    `linalg_ext.tile` is a 1-D loop construct that operates on tensors and
    evaluates its body once for each tile. The number and size of tiles is
    specified by the `tile_size` operand.

    The `tile` op takes a list of destination-passing style tensors and returns
    a matching list of tensors of the same size.

    Every instance of the body is expected to return a tile with leading
    dimension matching the corresponding tile size.

    The default terminator behavior is such that tiles yielded by individual
    iterations are concatenated along the `tiled_dim` dimension.
    This is the canonical way to perform "subset insertions".
    Note, if `tiled_dim` has the value `0`, it may be elided from pretty
    pinting and parsing.

    All return tiles are concatenated into forming the matching full result
    tensor according to the terminator.

    When the `tile_size` operand is a `tensor<..index>`, the `tile` op
    evaluates its body `dim(tile_size, 0)` times. Each iteration `i` produces a
    tile of leading size `tile_size[i]`.

    The induced `offset` block argument captures the running sum of `tile_size`
    for all the previous iterations.

    When the `tile_size` operand is a single index, it is interpreted as a
    sequence of tile sizes given by the following formula:
    ```
      N = tensor.dim(...)
      S = sizes
      T, R = divmod(N, S)
      [T] * S + ([R] if R != 0 else [])
    ```

    All tiles except the last are of the same size.
  }];
  let arguments = (ins AnyTypeOf<[// TODO: allow TensorOf<[Index]>,
                                  Index]>:$tile_size,
                       Variadic<AnyRankedTensor>:$outs,
                       I64Attr:$tiled_dim);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  let skipDefaultBuilders = 1;
  let builders = [
    // Builder that builds a tile on the implicit first dimension (i.e. `0`).
    OpBuilder<(ins "Value":$tileSizes, "ValueRange":$outs,
      CArg<"function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>",
           "nullptr">)>,
    // Builder that builds a tile with a specified integral dimension.
    OpBuilder<(ins "Value":$tileSizes, "ValueRange":$outs, "int64_t":$tiledDims,
      CArg<"function_ref<void(OpBuilder &, Location, Value, Value, ValueRange)>",
           "nullptr">)>,
  ];

  let extraClassDeclaration = [{
    static StringRef getTiledDimAttrName() { return "tiled_dim";}
    using TileOpBodyBuilderFn =
      function_ref<void(OpBuilder &, Location, Value /*offset*/, Value /*size*/,
                        ValueRange /*outs*/)>;
    // TODO: helper for getting named region args without magic constants etc.
  }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let verifier = [{ return ::verify(*this); }];
}

def LinalgExt_TileYieldOp : LinalgExt_PureOp<"tile_yield", [
    NoSideEffect, ReturnLike, Terminator]> {
  let summary = "LinalgExt yield op";
  let description = [{
    `linalg_ext.tile_yield` is a special terminator operation for blocks inside
    regions in `linalg_ext.tile`.
    The tiles yielded by individual iterations are concatenated along the first
    dimension. This is the canonical way to perform "subset insertions"
    (TODO: allow dim permutations).
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ /* nothing to do */ }]>,
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def LinalgExt_InParallelOp : LinalgExt_PureOp<"in_parallel", [
       SingleBlockImplicitTerminator<"linalg_ext::PerformConcurrentlyOp">,
       RecursiveSideEffects,
       AutomaticAllocationScope,
      ]> {
  let summary = "evaluate a block multiple times in parallel";
  let description = [{
    `linalg_ext.in_parallel` is a target-independent parallel function application
    operation. It has exactly one block that represents the parallel function body
    and it takes a single index operand that indicates how many parallel instances
    of that function should get instantiated.

    When the parallel function body is pure (i.e. has no side effects) then the only
    allowed terminator is `linalg_ext.perform_concurrently`, which dictates
    how the results of all parallel invocations should be reconciled into a full
    value that will be returned from `in_parallel`. Multi-value returns are encoded
    by including multiple operations inside the `perform_concurrently` block.

    When the parallel function body has side effects, the order of reads and writes
    to memory is unspecified across iterations.    

    This op resembles `scf.for` to a large degree, but crucially differs in that it
    (1) doesn't have `iter_args` and (2) has a special terminator, both of which
    enable reasoning about its parallel semantics. Another difference is that
    `in_parallel` always iterates over a range between 0 and an upper bound, but
    that's insignificant.
  }];
  let arguments = (ins Index:$num_threads);

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let verifier = [{ return ::verify(*this); }];

  // The default builder does not add the proper body BBargs, roll our own.
  let skipDefaultBuilders = 1;
  let builders = [
    // Bodyless builder, result types must be specified.
    OpBuilder<(ins "TypeRange":$resultTypes, "Value":$num_threads)>,
    // Builder that takes a bodyBuilder lambda, result types are inferred from 
    // the terminator.
    OpBuilder<(ins "Value":$num_threads,
              "function_ref<void(OpBuilder &, Location, Value)>":$bodyBuilder)>
  ];
  let extraClassDeclaration = [{
    Value getThreadIndex() { return getBody()->getArgument(0); }
    static void ensureTerminator(Region &region, Builder &builder, Location loc);
    PerformConcurrentlyOp getTerminator();
  }];
}

def LinalgExt_PerformConcurrentlyOp : LinalgExt_PureOp<"perform_concurrently", [
       NoSideEffect,
       Terminator,
       SingleBlockImplicitTerminator<"linalg_ext::EndPerformConcurrentlyOp">,
      ]> {
  let summary = "terminates a `in_parallel` block";
  let description = [{
    `linalg_ext.perform_concurrently` is a designated terminator for the blocks
    of `linalg_ext.in_parallel` operations. The terminator contains a single block
    itself, which describes how the results of each parallel invocation are to be
    reconciled into a single value to be returned from the parallel invocation.
    One operation in this terminator's block corresponds to a single return of
    `in_parallel`.
  }];

  let regions = (region SizedRegion<1>:$region);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let verifier = [{ return ::verify(*this); }];

  // TODO(apaszke, ntv): Add an interface for ops that can appear inside
  // perform_concurrently.
  let extraClassDeclaration = [{
    SmallVector<Type> yieldedTypes();
    SmallVector<ParallelInsertSliceOp> yieldingOps();
  }];
}

def LinalgExt_EndPerformConcurrentlyOp : LinalgExt_PureOp<"end_perform_concurrently", [
       NoSideEffect, Terminator]> {
  let summary = "terminates a `perform_concurrently` block";
  let description = [{
    A designated terminator for `perform_concurrently`. It's not expected to appear
    in the textual form of the IR.
  }];
}

def LinalgExt_ParallelInsertSliceOp : LinalgExt_PureOp<"parallel_insert_slice", [
       AttrSizedOperandSegments, OffsetSizeAndStrideOpInterface]> {
  let summary = "updates slices of a tensor concurrently";
  let description = [{
    Updates slices of a full tensor with multiple sub-slices concurrently.

    Conflicting writes result in undefined semantics, in that the indices written
    to by multiple parallel updates might contain data from any of the updates, or
    even a malformed bit pattern (in reality the semantics might end up depending
    on the memory model of the parallel hardware that `in_parallel` will be lowered to).

    If an index is updated by exactly one updates, the value contained at that index
    in the resulting tensor will be equal to the value at a corresponding index of a
    slice that was used for the updated. If an index is not updated at all, its value
    will be equal to the one in the original tensor.

    Note that we cannot mark this operation as pure (NoSideEffects), even
    though it has no side effects, because it will get DCEd during 
    canonicalization. Ideally we would use attributes instead of those funny
    terminating ops, but attributes cannot refer to SSA values at the moment, so
    it's the best we can do for now.
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    AnyRankedTensor:$dest,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    I64ArrayAttr:$static_offsets,
    I64ArrayAttr:$static_sizes,
    I64ArrayAttr:$static_strides
  );
  let assemblyFormat = [{
    $source `into` $dest ``
    custom<OperandsOrIntegersOffsetsOrStridesList>($offsets, $static_offsets)
    custom<OperandsOrIntegersSizesList>($sizes, $static_sizes)
    custom<OperandsOrIntegersOffsetsOrStridesList>($strides, $static_strides)
    attr-dict `:` type($source) `into` type($dest)
  }];

  let extraClassDeclaration = [{
    Type yieldedType() { return dest().getType(); }

    RankedTensorType getSourceType() {
      return source().getType().cast<RankedTensorType>();
    }

    /// Return the expected rank of each of the `static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before `offsets`, `sizes` and
    /// `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }
  }];

  let builders = [
    // Build a ParallelInsertSliceOp with mixed static and dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ArrayRef<OpFoldResult>":$offsets, "ArrayRef<OpFoldResult>":$sizes,
      "ArrayRef<OpFoldResult>":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>,
    // Build a ParallelInsertSliceOp with dynamic entries.
    OpBuilder<(ins "Value":$source, "Value":$dest,
      "ValueRange":$offsets, "ValueRange":$sizes, "ValueRange":$strides,
      CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
  ];

  let hasCanonicalizer = 1;
}

#endif // DIALECTS_LINALGEXT_LINALGEXTOPS
