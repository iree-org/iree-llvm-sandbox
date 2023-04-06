//===-- TupleOps.td - Tuple operations definitions ---------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TUPLE_IR_TUPLEOPS
#define TUPLE_IR_TUPLEOPS

include "iterators/Dialect/Tuple/IR/TupleDialect.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"

class Tuple_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tuple_Dialect, mnemonic, traits> {
}

def Tuple_FromElementsOp : Tuple_Op<"from_elements", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = "Builds a tuple from the given elements";
  let description = [{
    Returns a `tuple` with the elements assembled from the operands, i.e., the
    elements of the returned `tuple` consist of the values provided as operands.

    Example:

    ```mlir
    %0 = arith.constant 42 : i32
    %tuple = tuple.from_elements %0, %0 : tuple<i32, i32>  // contains (42, 42)
    ```
  }];
  let arguments = (ins Variadic<AnyType>:$elements);
  let results = (outs AnyTuple:$tuple);
  let assemblyFormat = [{
    $elements attr-dict `:` qualified(type($tuple))
    custom<TupleElementTypes>(type($elements), ref(type($tuple)))
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      setNameFn(getResult(), "tuple");
    }
  }];
}

def Tuple_ToElementsOp : Tuple_Op<"to_elements", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
    ]> {
  let summary = "Produces all elements of a given tuple";
  let description = [{
    Produces one result value for each field of the provided tuple. Note that
    this does not "flatten" the tuple, i.e., a tuple-typed element produces
    exactly one result value, which may itself have zero, one, or more elements.

    Example:

    ```mlir
    %0 = ...
    %first_element, %second_element = tuple.to_elements %0 : tuple<i32, tuple<>>
    // %first_element is a i32, %second_element is a tuple<>
    ```
  }];
  let arguments = (ins AnyTuple:$tuple);
  let results = (outs Variadic<AnyType>:$elements);
  let assemblyFormat = [{
    $tuple attr-dict `:` qualified(type($tuple))
    custom<TupleElementTypes>(type($elements), ref(type($tuple)))
  }];
  let extraClassDefinition = [{
    /// Implement OpAsmOpInterface.
    void $cppClass::getAsmResultNames(
        llvm::function_ref<void(mlir::Value, llvm::StringRef)> setNameFn) {
      if (getNumResults() > 0)
        setNameFn(getResult(0), "elements");
    }
  }];
}

#endif // TUPLE_IR_TUPLEOPS
