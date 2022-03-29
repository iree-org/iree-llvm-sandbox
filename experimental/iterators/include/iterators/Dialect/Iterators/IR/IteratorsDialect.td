//===-- IteratorsDialect.td - Iterators dialect ------------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Dialect definition
//===----------------------------------------------------------------------===//

def Iterators_Dialect : Dialect {
  let name = "iterators";
  let cppNamespace = "::mlir::iterators";
  let summary = "Dialect implementing database-style iterators.";
  let useDefaultTypePrinterParser = 1;
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT
