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
  let description = [{
    This dialect is designed to express computations on streams of data with the
    goal to minimize the number of transfers from slow to fast memory and back
    (e.g., from main memory to a CPU cache and back). Iterators are often also
    called [generators](https://en.wikipedia.org/wiki/Generator_(computer_programming)).

    The main data type for SSA values handled by this dialect are `Stream`s; the
    ops that consume and produce streams are called *iterators* or *iterator
    ops*. The iterators producing the operand `Stream`s are called *upstream*
    iterators; those that consume the result `Stream` are called *downstream*
    iterators.

    Example:

    ```mlir
    %input = "iterators.constantstream"()
               { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] } :
               () -> (!iterators.stream<!element_type>)
    "iterators.sink"(%input) : (!iterators.stream<!T>) -> ()
    ```

    The semantics of the different iterator ops is defined as if execution was
    done one iterator at the time, i.e., as if the input streams of the operands
    of each iterator existed in their entirety before the op started executing,
    and that that op ran from start to finish atomically producing the result
    stream in its entirety. However, the intention is to lower a program of
    iterator ops such that **all iterators** in a connected component of
    iterators **run at the same time**, continuously passing individual elements
    of the streams (and potentially control-flow) between them. In many cases,
    this removes the need to write the potentially large result of one iterator
    (represented by a stream) to slow memory because each element in the stream
    is consumed by the subsequent (downstream) iterator immediately after it has
    been produced.
  }];
  let useDefaultTypePrinterParser = 1;
}

#endif // ITERATORS_DIALECT_ITERATORS_IR_ITERATORSDIALECT
