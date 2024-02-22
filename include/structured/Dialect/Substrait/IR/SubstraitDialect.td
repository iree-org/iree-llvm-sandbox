//===-- SubstraitDialect.td - Substrait dialect ------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITDIALECT
#define SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Substrait_Dialect : Dialect {
  let name = "substrait";
  let cppNamespace = "::mlir::substrait";
  let summary = "Dialect for representing Substrait plans in MLIR.";
  let description = [{
    This dialect is intented to represent [Substrait](https://substrait.io/)
    query plans for relational algebra in MLIR. This may be useful for both
    producers and consumers, which may use this dialect as a result or input
    dialect of an MLIR-based pipeline, respectively, or for implementing
    rewrites on Substrait that optimize or legalize plans between existing
    Substraits producers and consumers.

    The ops and types in this dialect have an approximate one-to-one
    correspondance with the specification and the
    [protobuf](https://github.com/substrait-io/substrait/tree/main/proto/substrait)
    message types. The correspondance is only approximate since it is often
    more natural in MLIR to represent several message types as a single op and
    express message sub-types with interfaces instead.
  }];
}

#endif // SUBSTRAIT_DIALECT_SUBSTRAIT_IR_SUBSTRAITDIALECT
