//===-- IteratorsInterfaces.td - Iterators interfaces -------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_ITERATORS_IR_ITERATORSINTERFACES
#define STRUCTURED_DIALECT_ITERATORS_IR_ITERATORSINTERFACES

include "mlir/IR/OpBase.td"

def Iterators_IteratorOpInterface : OpInterface<"IteratorOpInterface"> {
  let description = [{Iterators op interface.}];
}

#endif // STRUCTURED_DIALECT_ITERATORS_IR_ITERATORSINTERFACES
