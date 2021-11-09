//===-- LinalgExtOps.td - Linalg Extension dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_LINALGEXT_LINALGEXTOPS
#define RUNNERS_LINALGEXT_LINALGEXTOPS

include "include/LinalgExt/LinalgExtBase.td"
include "include/LinalgExt/LinalgExtInterfaces.td"
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

#endif // RUNNERS_LINALGEXT_LINALGEXTOPS
