//===-- DataFlowOps.td - DataFlow operations definitions ---*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWOPS
#define ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWOPS

include "iterators/Dialect/DataFlow/IR/DataFlowDialect.td"
include "iterators/Dialect/DataFlow/IR/DataFlowTypes.td"
include "mlir/IR/OpBase.td"

class DataFlow_Op<string mnemonic, list<Trait> traits = []> :
    Op<DataFlow_Dialect, mnemonic, traits> {
}

def DataFlow_SampleInputOp : DataFlow_Op<"sampleInput"> {
  let summary = "Create some sample input";
  let results = (outs DataFlow_Stream);
}

def DataFlow_ReduceOp : DataFlow_Op<"reduce"> {
  let summary = "Reduce the input to a single tuple";
  let arguments = (ins DataFlow_Stream);
  let results = (outs DataFlow_Stream);
}

def DataFlow_SinkOp : DataFlow_Op<"sink"> {
  let summary = "Consume tuples from an iterator";
  let arguments = (ins DataFlow_Stream);
}

#endif // ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWOPS
