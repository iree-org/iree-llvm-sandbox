//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_CONVERSION_PASSES
#define ITERATORS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// IteratorsToLLVM
//===----------------------------------------------------------------------===//

def ConvertIteratorsToLLVM : Pass<"convert-iterators-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations from the iterators dialect into the "
                "LLVM dialect";
  let description = [{
    This lowering pass converts iterators, i.e., operation that have operands
    and results of type `Stream`, to control-flow that co-executes all iterators
    connected via use-def chains of `Stream`s as well as the state that is
    required for each iterator to do so. This is achieved by having each
    iterator *consumes* the elements in its operand `Stream`s in order to
    produce the elements of its result `Stream`. Since consuming an element is
    destructive (i.e., each element can only be consumed once), the lowering
    currently only works if the use-def chains of `Stream`s form a tree, i.e.,
    every `Stream` is used as an operand by exactly one subsequent iterator.

    More precisely, for each iterator, the lowering produces a state with a
    number of typed fields, including any local state that the iterator might
    require **plus the states of all iterators in the transitive use-def chain**
    of its operands. The computations are expressed as three functions, `Open`,
    `Next`, and `Close`, which operate on that state and which continuously pass
    control-flow between the logic of the different iterators:

    * `Open` initializes the computations, typically calling `Open` on the
      nested states of the current iterator;
    * `Next` produces the next element in the stream or signals "end of stream",
      making zero, one, or more calls to `Next` on any of the nested states as
      required by the logic of the current iterator; and
    * `Close` cleans up the state if necessary, typically calling `Close` on the
      nested states of the current iterator.

    The three functions take the current iterator state as an input and return
    the updated state. (Subsequent bufferization within LLVM presumably converts
    this to in-place updates.) `Next` also returns the next element in the
    stream, plus a Boolean that signals whether the element is valid or the end
    of the stream was reached.
  }];
  let constructor = "mlir::createConvertIteratorsToLLVMPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "LLVM::LLVMDialect",
    "scf::SCFDialect"
  ];
}

//===----------------------------------------------------------------------===//
// IteratorsToLLVM
//===----------------------------------------------------------------------===//

def ConvertStatesToLLVM : Pass<"convert-states-to-llvm", "ModuleOp"> {
  let summary = "Convert the operations on iterator states into the LLVM "
                "dialect";
  let description = [{
    This lowering pass converts operations on iterator states into equivalent
    operations of the LLVM dialect. Currently, the ops on iterator states are
    essentially equivalent to the LLVM ops dealing with structs (but allow
    arbitrary types), so the lowering only consists of straightforward,
    one-to-one patterns.
  }];
  let constructor = "mlir::createConvertStatesToLLVMPass()";
  let dependentDialects = [
    "LLVM::LLVMDialect"
  ];
}

//===----------------------------------------------------------------------===//
// TabularToLLVM
//===----------------------------------------------------------------------===//

def ConvertTabularToLLVM : Pass<"convert-tabular-to-llvm", "ModuleOp"> {
  let summary = "Convert the tabular dialect to the LLVM dialect";
  let description = [{
    Convert the data types and ops from the tabular dialect to the LLVM dialect.
    The data types are typically variations of LLVM's structs and pointers; the
    ops, thus, translate to the LLVM ops handling structs and pointers.
  }];
  let constructor = "mlir::createConvertTabularToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // ITERATORS_CONVERSION_PASSES
