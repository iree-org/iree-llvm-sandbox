//===-- DataFlowDialect.td - DataFlow dialect --------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWDIALECT
#define ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def DataFlow_Dialect : Dialect {
  let name = "dataflow";
  let cppNamespace = "::mlir::dataflow";
  let summary = "Dialect implementing database-style dataflow operators.";
  let useDefaultTypePrinterParser = 1;
}

#endif // ITERATORS_DIALECT_DATAFLOW_IR_DATAFLOWDIALECT
