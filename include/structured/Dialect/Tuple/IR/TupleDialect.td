//===-- TupleDialect.td - Tuple dialect --------------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TUPLE_IR_TUPLEDIALECT
#define TUPLE_IR_TUPLEDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition.
//===----------------------------------------------------------------------===//

def Tuple_Dialect : Dialect {
  let name = "tuple";
  let cppNamespace = "::mlir::structured";
  let summary = "Dialect for generic manipulation of built-in tuples.";
  let description = [{
    This dialect contains ops that allow creating tuples as well as accessing
    and updating elements or slices of them. Potential lowering are a mapping to
    lower-level tuple-like structures such as structs in LLVM or decomposition
    into individual elements.
  }];
}

#endif // TUPLE_IR_TUPLEDIALECT
