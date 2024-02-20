//===-- SubstraitInterfaces.td - Substrait interfaces ------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITINTERFACES
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITINTERFACES

include "mlir/IR/OpBase.td"

def Substrait_RelOpInterface : OpInterface<"RelOpInterface"> {
  let description = [{
    Interface for any relational operation in a Substrait plan. This corresponds
    to a `Rel` message, which only consists of the `rel_type` field, which, in
    turn, holds a more specialized message with the information specific to the
    concrete relational operation.
  }];
  let cppNamespace = "::mlir::substrait";
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITINTERFACES
