//===-- IteratorsOps.td - Iterators operations definitions -*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS

include "iterators/Dialect/Iterators/IR/IteratorsDialect.td"
include "iterators/Dialect/Iterators/IR/IteratorsTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpBase.td"

class Iterators_Op<string mnemonic, list<Trait> traits = []> :
    Op<Iterators_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// Debugging/testing utilities
//===----------------------------------------------------------------------===//

def MatchingFieldCountsPred
    : CPred<[{$tuple.getType().dyn_cast<TupleType>().size() ==
              $values.dyn_cast<ArrayAttr>().size()}]>;
def MatchingFieldCountsConstraint
    : PredOpTrait<"number of values matches tuple size",
                  MatchingFieldCountsPred>;

def Iterators_ConstantTupleOp
    : Iterators_Op<"constant", [MatchingFieldCountsConstraint]> {
  let summary = "Creates a tuple from the given values";
  // TODO(ingomueller): extend to more field types
  let arguments = (ins I32ArrayAttr:$values);
  let results = (outs TupleOf<[I32]>:$tuple);
}

def Iterators_PrintTupleOp : Iterators_Op<"print"> {
  let summary = "Print the elements of a tuple";
  // TODO(ingomueller): extend to all supported tuple types
  let arguments = (ins TupleOf<[I32]>);
}

//===----------------------------------------------------------------------===//
// High-level iterators
//===----------------------------------------------------------------------===//

def Iterators_SampleInputOp : Iterators_Op<"sampleInput"> {
  let summary = "Create some sample input";
  let results = (outs Iterators_Stream:$result);
}

def Iterators_ReduceOp : Iterators_Op<"reduce"> {
  let summary = "Reduce the input to a single tuple";
  let arguments = (ins Iterators_Stream:$input);
  let results = (outs Iterators_Stream:$result);
}

def Iterators_SinkOp : Iterators_Op<"sink"> {
  let summary = "Consume tuples from an iterator";
  let arguments = (ins Iterators_Stream:$input);
}

//===----------------------------------------------------------------------===//
// Create/Open/Next/Close interface of low-level iterators
//===----------------------------------------------------------------------===//

def Iterators_OpenOp : Iterators_Op<"open", [SameOperandsAndResultType]> {
  let summary = "Opens the given operator";
  let arguments = (ins AnyIteratorState:$inputState);
  let results = (outs AnyIteratorState:$resultState);
}

def Iterators_NextOp : Iterators_Op<"next",
    [AllTypesMatch<["inputState", "resultState"]>,
     TypesMatchWith<"returned element matches element type of state",
                    "resultState", "nextElement",
                    "$_self.cast<IteratorInterface>().getElementType()">
    ]> {
  let summary = "Consumes the next element from the given operator";
  let arguments = (ins AnyIteratorState:$inputState);
  let results = (outs AnyIteratorState:$resultState, I1:$hasNext, AnyType:$nextElement);
}

def Iterators_CloseOp : Iterators_Op<"close", [SameOperandsAndResultType]> {
  let summary = "Closes the given operator";
  let arguments = (ins AnyIteratorState:$inputState);
  let results = (outs AnyIteratorState:$resultState);
}

//===----------------------------------------------------------------------===//
// State creation of low-level iterators
//===----------------------------------------------------------------------===//

def Iterators_CreateSampleInputStateOp : Iterators_Op<"createSampleInputState"> {
  let summary = "Create initial state of SampleInput iterator";
  let results = (outs Iterators_SampleInputState:$createdState);
  let hasVerifier = true;
}

def Iterators_CreateReduceStateOp : Iterators_Op<"createReduceState"> {
  let summary = "Create initial state of Reduce iterator";
  let arguments = (ins AnyIteratorState:$upstreamState);
  let results = (outs Iterators_ReduceState:$createdState);
  let hasVerifier = true;
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS
