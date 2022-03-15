//===- IteratorsOps.td - Iterators operations definitions --*- tablegen -*-===//

#ifndef DIALECTS_ITERATORS_ITERATORSOPS
#define DIALECTS_ITERATORS_ITERATORSOPS

include "Dialects/Iterators/IteratorsDialect.td"
include "Dialects/Iterators/IteratorsTypes.td"

class Iterators_Op<string mnemonic, list<Trait> traits = []> :
    Op<Iterators_Dialect, mnemonic, traits> {
}

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

#endif // DIALECTS_ITERATORS_ITERATORSOPS
