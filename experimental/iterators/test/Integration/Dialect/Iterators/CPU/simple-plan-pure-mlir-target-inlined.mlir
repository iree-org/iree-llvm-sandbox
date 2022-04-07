//===-- simple-plan-pure-mlir-target.mlir ----------------------*- mlir -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains a manual translation of the "simple plan" (see
/// simple-plan-e2e.mlir) to a potential new conversion target, which only
/// consists of MLIR (SCF, Func, and LLVM dialects) rather than function calls
/// into a C/C++ implementation that does all of the heavy lifting. In this
/// version, the Create/Open/Next/Close interface is done via values (i.e., the
/// function calls are inlined).
///
//===----------------------------------------------------------------------===//
//
// RUN: mlir-opt -convert-scf-to-cf -convert-func-to-llvm %s | \
// RUN:   mlir-cpu-runner -e main -entry-point-result=void | \
// RUN:   FileCheck %s

/// Example tuple type. Tuple types vary across and within queries and, thus,
/// have to be generated.
!intTuple = type !llvm.struct<(i32)>

/// Optional for signaling "end-of-stream". First field is true iff the
/// optional carries a valid value.
!optionalIntTuple = type !llvm.struct<(i1, !intTuple)>

/// State of the sample intput operator (consisting of currentIndex and
/// lastIndex).
!sampleInputOperatorState = type !llvm.struct<(i32, i32)>

/// State of the reduce operator (consisting of its upstream operator). Since
/// this operator should be able to consume from any upstream operator, the
/// exact type of the state depends on that upstream operator and, thus, has to
/// be generated.
!reduceOperatorState = type !llvm.struct<(!sampleInputOperatorState)>

module {
  //===--------------------------------------------------------------------===//
  // Tuples and optional
  //===--------------------------------------------------------------------===//

  // All of these depend on the tuple type, so their instantiations need to be
  // generated for each required tuple type.

  func private @makeTupleFromValue(%value: i32) -> (!intTuple) {
    %emptyTuple = llvm.mlir.undef : !intTuple
    %ret = llvm.insertvalue %value, %emptyTuple[0] : !intTuple
    return %ret : !intTuple
  }

  func private @makeEmptyOptional() -> !optionalIntTuple {
    %emptyOptional = llvm.mlir.undef : !optionalIntTuple
    %false = arith.constant 0 : i1
    %ret = llvm.insertvalue %false, %emptyOptional[0] : !optionalIntTuple
    return %ret : !optionalIntTuple
  }

  func private @makeOptionalFromTuple(%tuple: !intTuple) -> !optionalIntTuple {
    %emptyOptional = llvm.mlir.undef : !optionalIntTuple
    %true = arith.constant 1 : i1
    %optionalWithBool = llvm.insertvalue %true, %emptyOptional[0] : !optionalIntTuple
    %finalOptional = llvm.insertvalue %tuple, %optionalWithBool[1] : !optionalIntTuple
    return %finalOptional : !optionalIntTuple
  }

  //===--------------------------------------------------------------------===//
  // Printing tuples and optionals
  //===--------------------------------------------------------------------===//

  llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
  llvm.mlir.global internal constant @printfFmtString("(%d)\0A\00")
  llvm.mlir.global internal constant @printfFmtStringEmpty("()\0A\00")

  /// Prints a tuple as "(val)".
  func private @iteratorsPrintTuple(%tuple: !intTuple) {
    %zero = llvm.mlir.constant(0 : i32) : i32
    %fmtStringAddr = llvm.mlir.addressof @printfFmtString : !llvm.ptr<array<6 x i8>>
    %fmtString = llvm.getelementptr %fmtStringAddr[%zero, %zero] : (!llvm.ptr<array<6 x i8>>, i32, i32) -> !llvm.ptr<i8>
    %value = llvm.extractvalue %tuple[0] : !intTuple
    llvm.call @printf(%fmtString, %value) : (!llvm.ptr<i8>, i32) -> i32
    return
  }

  /// Prints an optional tuple as "(val)" or "()".
  func private @iteratorsPrintOptionalTuple(%maybeTuple: !optionalIntTuple) {
    %hasValue = llvm.extractvalue %maybeTuple[0] : !optionalIntTuple
    scf.if %hasValue -> () {
      %tuple = llvm.extractvalue %maybeTuple[1] : !optionalIntTuple
      call @iteratorsPrintTuple(%tuple) : (!intTuple) -> ()
    } else {
      %zero = llvm.mlir.constant(0 : i32) : i32
      %fmtStringAddr = llvm.mlir.addressof @printfFmtStringEmpty : !llvm.ptr<array<4 x i8>>
      %fmtString = llvm.getelementptr %fmtStringAddr[%zero, %zero] : (!llvm.ptr<array<4 x i8>>, i32, i32) -> !llvm.ptr<i8>
      llvm.call @printf(%fmtString) : (!llvm.ptr<i8>) -> i32
    }
    return
  }

  //===--------------------------------------------------------------------===//
  // Sum
  //===--------------------------------------------------------------------===//

  /// Sums up the values of two single-int tuples. This is a sample reduce
  /// reduce function, by which the reduce operator could be parametrized.
  func private @iteratorsSum(%lhs: !intTuple, %rhs: !intTuple) -> !intTuple {
    %lhsInt = llvm.extractvalue %lhs[0] : !intTuple
    %rhsInt = llvm.extractvalue %rhs[0] : !intTuple
    %sum = arith.addi %lhsInt, %rhsInt : i32
    %emptyTuple = llvm.mlir.undef : !intTuple
    %res = llvm.insertvalue %sum, %emptyTuple[0] : !intTuple
    return %res : !intTuple
  }

  //===--------------------------------------------------------------------===//
  // Query
  //===--------------------------------------------------------------------===//
  func @main() {
    // Sample input operator -------------------------------------------------//

    // Create initial state of input operator.
    // [2] Inlined iteratorsSampleInputOperatorCreate begin.
    %emptyInputState = llvm.mlir.undef : !sampleInputOperatorState

    // Initialize current index.
    %zero = arith.constant 0 : i32
    %stateWithCurrent = llvm.insertvalue %zero, %emptyInputState[0] : !sampleInputOperatorState

    // Initialize last index.
    %three = arith.constant 3 : i32
    %initialInputState = llvm.insertvalue %three, %stateWithCurrent[1] : !sampleInputOperatorState
    // [2] Inlined iteratorsSampleInputOperatorCreate end.

    // Reduce operator -------------------------------------------------------//

    // [1] Inlined iteratorsReduceOperatorCreate begin.
    %emptyReduceState = llvm.mlir.undef : !reduceOperatorState
    %initialReduceState = llvm.insertvalue %initialInputState, %emptyReduceState[0] : !reduceOperatorState
    // [1] Inlined iteratorsReduceOperatorCreate end.

    // Print and consume -----------------------------------------------------//

    // [0] Inlined iteratorsConsumeAndPrint begin.

    // [1]   Inlined iteratorsReduceOperatorOpen begin.
    %upstreamState = llvm.extractvalue %initialReduceState[0] : !reduceOperatorState

    // [2]     Inlined iteratorsSampleInputOperatorOpen begin.
    %updatedUpstreamState = llvm.insertvalue %zero, %upstreamState[0] : !sampleInputOperatorState
    // [2]     Inlined iteratorsSampleInputOperatorOpen end.

    %openedState = llvm.insertvalue %updatedUpstreamState, %initialReduceState[0] : !reduceOperatorState
    // [1]   Inlined iteratorsReduceOperatorOpen end.

    %emptyTuple, %consumedState = scf.while(%currentReduceState = %openedState) : (!reduceOperatorState) -> (!optionalIntTuple, !reduceOperatorState) {
      // [1]   Inlined iteratorsReduceOperatorNext end.
      %currentInputState = llvm.extractvalue %currentReduceState[0] : !reduceOperatorState

      // [2]     Consume first tuple.
      // [3]       Inlined iteratorsSampleInputOperatorNext beign.
      %firstCurrentIndex = llvm.extractvalue %currentInputState[0] : !sampleInputOperatorState
      %firstLastIndex = llvm.extractvalue %currentInputState[1] : !sampleInputOperatorState
      %firstHasNext = arith.cmpi "slt", %firstCurrentIndex, %firstLastIndex : i32

      %maybeFirstTuple, %upstreamStateAfterFirstNext = scf.if %firstHasNext -> (!optionalIntTuple, !sampleInputOperatorState) {
        // [3]       Update state (i.e., increment current index)
        %one = arith.constant 1 : i32
        %nextIndex = arith.addi %firstCurrentIndex, %one : i32
        %updatedState = llvm.insertvalue %nextIndex, %currentInputState[0] : !sampleInputOperatorState

        // [3]       Make optional tuple from value at current index.
        %tuple = call @makeTupleFromValue(%nextIndex) : (i32) -> !intTuple
        %ret = call @makeOptionalFromTuple(%tuple) : (!intTuple) -> !optionalIntTuple

        scf.yield %ret, %updatedState : !optionalIntTuple, !sampleInputOperatorState
      } else {
        // [3]       Done iterating over sample data --> return empty optional.
        %ret = call @makeEmptyOptional() : () -> !optionalIntTuple
        scf.yield %ret, %currentInputState : !optionalIntTuple, !sampleInputOperatorState
      }
      // [3]       Inlined iteratorsSampleInputOperatorNext end.

      %firstTupleHasValue = llvm.extractvalue %maybeFirstTuple[0] : !optionalIntTuple

      // [2]     Run main loop of consuming upstream.
      %maybeTuple, %consumedUpstreamState = scf.if %firstTupleHasValue -> (!optionalIntTuple, !sampleInputOperatorState) {
        %firstTuple = llvm.extractvalue %maybeFirstTuple[1] : !optionalIntTuple

        %emptyOptional, %agg, %consumedUpstreamState = scf.while (%agg = %firstTuple, %currentUpstreamState = %upstreamStateAfterFirstNext) : (!intTuple, !sampleInputOperatorState) -> (!optionalIntTuple, !intTuple, !sampleInputOperatorState) {
          // [3]       Inlined iteratorsSampleInputOperatorNext begin.
          %currentIndex = llvm.extractvalue %currentUpstreamState[0] : !sampleInputOperatorState
          %lastIndex = llvm.extractvalue %currentUpstreamState[1] : !sampleInputOperatorState
          %hasNext = arith.cmpi "slt", %currentIndex, %lastIndex : i32

          %maybeTuple, %nextUpstreamState = scf.if %hasNext -> (!optionalIntTuple, !sampleInputOperatorState) {
            // [3]       Update state (i.e., increment current index)
            %one = arith.constant 1 : i32
            %nextIndex = arith.addi %currentIndex, %one : i32
            %updatedState = llvm.insertvalue %nextIndex, %currentUpstreamState[0] : !sampleInputOperatorState

            // [3]       Make optional tuple from value at current index.
            %tuple = call @makeTupleFromValue(%nextIndex) : (i32) -> !intTuple
            %ret = call @makeOptionalFromTuple(%tuple) : (!intTuple) -> !optionalIntTuple

            scf.yield %ret, %updatedState : !optionalIntTuple, !sampleInputOperatorState
          } else {
            // [3]       Done iterating over sample data --> return empty optional.
            %ret = call @makeEmptyOptional() : () -> !optionalIntTuple
            scf.yield %ret, %currentUpstreamState : !optionalIntTuple, !sampleInputOperatorState
          }
          // [3]       Inlined iteratorsSampleInputOperatorNext end.

          %hasValue = llvm.extractvalue %maybeTuple[0] : !optionalIntTuple
          scf.condition(%hasValue) %maybeTuple, %agg, %nextUpstreamState : !optionalIntTuple, !intTuple, !sampleInputOperatorState
        } do {
        ^bb0(%maybeTuple: !optionalIntTuple, %agg: !intTuple, %nextUpstreamState: !sampleInputOperatorState):
          %tuple =  llvm.extractvalue %maybeTuple[1] : !optionalIntTuple
          %newAgg = call @iteratorsSum(%agg, %tuple) : (!intTuple, !intTuple) -> (!intTuple)
          scf.yield %newAgg, %nextUpstreamState : !intTuple, !sampleInputOperatorState
        }

        // [2]     Return aggregate wrapped in optional.
        %retTuple = call @makeOptionalFromTuple(%agg) : (!intTuple) -> !optionalIntTuple
        scf.yield %retTuple, %consumedUpstreamState : !optionalIntTuple, !sampleInputOperatorState
      } else {
        // [2]     Upstream provided no tuple --> return empty optional.
        %retTuple = call @makeEmptyOptional() : () -> !optionalIntTuple
        scf.yield %retTuple, %upstreamStateAfterFirstNext : !optionalIntTuple, !sampleInputOperatorState
      }

      // [2]     Update upstream state.
      %nextState = llvm.insertvalue %consumedUpstreamState, %currentReduceState[0] : !reduceOperatorState
      // [1]   Inlined iteratorsReduceOperatorNext end.

      %hasValue = llvm.extractvalue %maybeTuple[0] : !optionalIntTuple
      scf.condition(%hasValue) %maybeTuple, %nextState : !optionalIntTuple, !reduceOperatorState
    } do {
    ^bb0(%maybeTuple: !optionalIntTuple, %nextState: !reduceOperatorState):
      %tuple = llvm.extractvalue %maybeTuple[1] : !optionalIntTuple
      call @iteratorsPrintTuple(%tuple) : (!intTuple) -> ()
      scf.yield %nextState : !reduceOperatorState
    }
    // CHECK: (6)

    // [1]   Inlined iteratorsReduceOperatorClose begin.
    %consumedUpstreamState = llvm.extractvalue %consumedState[0] : !reduceOperatorState

    // [2]     Inlined iteratorsSampleInputOperatorClose begin.
    // [3]       iteratorsSampleInputOperatorClose is a no-op
    // [2]     Inlined iteratorsSampleInputOperatorClose end.

    %finalState = llvm.insertvalue %consumedUpstreamState, %consumedState[0] : !reduceOperatorState
    // [1]   Inlined iteratorsReduceOperatorClose end.

    // [0] Inlined iteratorsConsumeAndPrint end.

    return
  }
}
