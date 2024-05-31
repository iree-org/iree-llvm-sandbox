//===-- Passes.td - Substrait pass definition file ---------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef STRUCTURED_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES
#define STRUCTURED_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SubstraitEmitDeduplicationPass
    : Pass<"substrait-emit-deduplication"> {
  let summary = "Remove duplicate emit fields.";
  let description = [{
    Removes duplicates in the mapping of `emit` ops. This is somewhat similar to
    CSE in that it fuses redundant values; however, the redudant values are
    fields in the tuples/rows/structs inside of a `Relation` rather than SSA
    values. The deduplication consist of a pattern for each `RelOp` that removes
    the duplicate fields in a preceeding `emit` op and then re-establishes the
    original sequence of fields with a subsequent `emit` op (which may be fused
    with other emit ops and/or enable further deduplication).

    Example:

    ```mlir
    %0 = ...
    %1 = ...
    %2 = emit [0, 0] from %0 : tuple<si32> -> tuple<si32, si32>
    %3 = cross %1 x %2 : tuple<si1> x tuple<si32, si32>
    yield $3 : tuple<si1, si32, si32>
    ```

    Here, the `emit` op introduces a duplicate field by emitting the field `0`
    twice, so subsequent `RelOp`s have larger inputs than necessary. The pass
    pushes the duplication through the subsequent op, `cross`, like this:

    ```mlir
    %0 = ...
    %1 = ...
    %2 = emit [0] from %0 : tuple<si32> -> tuple<si32, si32>
    %3 = cross %1 x %2 : tuple<si1> x tuple<si32>
    %4 = emit [0, 1, 1] from %0 : tuple<si1, si32> -> tuple<si1, si32, si32>
    yield $4 : tuple<si1, si32, si32>
    ```

    The final `emit` cannot be pushed further as the encompassing `relation`
    needs to keep the fields indicated by the `yield` op.
  }];
  let constructor = "::mlir::substrait::createEmitDeduplicationPass()";
}

#endif // STRUCTURED_DIALECT_SUBSTRAIT_TRANSFORMS_PASSES
