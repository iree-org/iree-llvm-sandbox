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
// Iterators
//===----------------------------------------------------------------------===//

def Iterators_SampleInputOp : Iterators_Op<"sampleInput"> {
  let summary = "Create some sample input";
  let results = (outs Iterators_Iterator);
}

def Iterators_ReduceOp : Iterators_Op<"reduce"> {
  let summary = "Reduce the input to a single tuple";
  let arguments = (ins Iterators_Iterator);
  let results = (outs Iterators_Iterator);
}

def Iterators_SinkOp : Iterators_Op<"sink"> {
  let summary = "Consume tuples from an iterator";
  let arguments = (ins Iterators_Iterator);
}

//===----------------------------------------------------------------------===//
// Optional
//===----------------------------------------------------------------------===//

def Iterators_OptionalEmptyOp : Iterators_Op<"emptyoptional"> {
  let summary = "Creates an empty optional";
  let results = (outs Iterators_Optional);
}

def Iterators_OptionalInsertValueOp : Iterators_Op<"insertvalue"> {
  let summary = "Inserts a value into an optional (and sets the valid bit)";
  let arguments = (ins Iterators_Optional:$optional, AnyType:$value);
  let results = (outs Iterators_Optional:$result);
  let hasVerifier = true;
}

def Iterators_OptionalExtractValueOp : Iterators_Op<"extractvalue"> {
  let summary = "Extracts the value of a non-empty optional";
  let arguments = (ins Iterators_Optional:$input);
  let results = (outs AnyType:$result);
  let hasVerifier = true;
}

def Iterators_OptionalHasValueOp : Iterators_Op<"hasvalue"> {
  let summary = "Extracts the valid bit of an optional";
  let arguments = (ins Iterators_Optional);
  let results = (outs I1);
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSOPS
