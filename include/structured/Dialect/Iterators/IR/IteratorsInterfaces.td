//===-- IteratorsInterfaces.td - Iterators interfaces -------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_DIALECT_ITERATORS_IR_ITERATORSINTERFACES
#define ITERATORS_DIALECT_ITERATORS_IR_ITERATORSINTERFACES

include "mlir/IR/OpBase.td"

def Iterators_IteratorOpInterface : OpInterface<"IteratorOpInterface"> {
  let description = [{Iterators op interface.}];
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSINTERFACES
