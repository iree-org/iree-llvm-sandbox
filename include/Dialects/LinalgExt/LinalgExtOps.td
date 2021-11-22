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

class LinalgExt_PureOp<string mnemonic, list<OpTrait> traits = []> :
    Op<LinalgExt_Dialect, mnemonic, traits> {
}

class LinalgExt_Op<string mnemonic, list<OpTrait> traits = []> :
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
        "getLoopBounds",
        "getTiledImplementation"]>]> {
  let summary = "Reverse operator";
  let description = [{TODO}];

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
    specified by the `tile_sizes` operand.

    The `tile` op takes a list of destination-passing style tensors and returns
    a matching list of tensors of the same size.

    Every instance of the body is expected to return a tile with leading
    dimension matching the corresponding tile size.

    The default terminator behavior is such that tiles yielded by individual
    iterations are concatenated along the first dimension. This is the canonical
    way to perform "subset insertions".

    All return tiles are concatenated into forming the matching full result
    tensor according to the terminator.

    In the future, a "dim" permutation attribute is expected to be added to the
    terminator to modify the behavior of tile concatenation into the result.

    When the `tile_sizes` operand is a `tensor<..index>`, the `tile` op
    evaluates its body `dim(tile_sizes, 0)` times. Each iteration `i` produces a
    tile of leading size `tile_sizes[i]`.

    The induced `offset` block argument captures the running sum of `tile_sizes`
    for all the previous iterations.

    When the `tile_sizes` operand is a single index, it is interpreted as a
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
                                  Index]>:$tile_sizes,
                       Variadic<AnyRankedTensor>:$outs);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);
  // TODO: Placeholder to properly implement when needed.
  // let skipDefaultBuilders = 1;
  // let builders = [
  //   OpBuilder<(ins "Value":$tile_sizes,
  //     CArg<"function_ref<void(OpBuilder &, Location, /**/Value, Value)>",
  //          "nullptr">)>
  // ];

  let extraClassDeclaration = [{
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

#endif // DIALECTS_LINALGEXT_LINALGEXTOPS
