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

class Iterators_Op<string mnemonic, list<Trait> traits = []> :
    Op<Iterators_Dialect, mnemonic, traits> {
}

//===----------------------------------------------------------------------===//
// High-level iterators
//===----------------------------------------------------------------------===//

def Iterators_SampleInputOp : Iterators_Op<"sampleInput"> {
  let summary = "Create some sample input";
  let results = (outs Iterators_Stream);
}

def Iterators_ReduceOp : Iterators_Op<"reduce"> {
  let summary = "Reduce the input to a single tuple";
  let arguments = (ins Iterators_Stream);
  let results = (outs Iterators_Stream);
}

def Iterators_SinkOp : Iterators_Op<"sink"> {
  let summary = "Consume tuples from an iterator";
  let arguments = (ins Iterators_Stream);
}

//===----------------------------------------------------------------------===//
// Create/Open/Next/Close interface of low-level iterators
//===----------------------------------------------------------------------===//

def Iterators_OpenOp : Iterators_Op<"open"> {
  let summary = "Opens the given operator";
  let arguments = (ins AnyIteratorState:$inputState);
  let results = (outs AnyIteratorState:$resultState);
  let hasVerifier = true;
}

def Iterators_NextOp : Iterators_Op<"next"> {
  let summary = "Consumes the next element from the given operator";
  let arguments = (ins AnyIteratorState:$inputState);
  let results = (outs AnyIteratorState:$resultState, I1:$hasNext, AnyType:$nextElement);
  let hasVerifier = true;
}

def Iterators_CloseOp : Iterators_Op<"close"> {
  let summary = "Closes the given operator";
  let arguments = (ins AnyIteratorState:$inputState);
  let results = (outs AnyIteratorState:$resultState);
  let hasVerifier = true;
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
