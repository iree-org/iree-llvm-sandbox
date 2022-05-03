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
/// version, the Create/Open/Next/Close interface is done via function calls.
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
  // Sample input operator
  //===--------------------------------------------------------------------===//

  /// Creates an initial state of the sample input operator (which generates the
  /// sequence of ints [1, 2, 3]).
  func private @iteratorsSampleInputOperatorCreate() -> !sampleInputOperatorState {
    %emptyState = llvm.mlir.undef : !sampleInputOperatorState

    // Initialize current index.
    %zero = arith.constant 0 : i32
    %stateWithCurrent = llvm.insertvalue %zero, %emptyState[0] : !sampleInputOperatorState

    // Initialize last index.
    %three = arith.constant 3 : i32
    %completeState = llvm.insertvalue %three, %stateWithCurrent[1] : !sampleInputOperatorState

    return %completeState : !sampleInputOperatorState
  }

  /// "Opens" the iterator (i.e., sets the current index to 0).
  func private @iteratorsSampleInputOperatorOpen(%state: !sampleInputOperatorState) -> !sampleInputOperatorState {
    %zero = arith.constant 0 : i32
    %updatedState = llvm.insertvalue %zero, %state[0] : !sampleInputOperatorState
    return %updatedState : !sampleInputOperatorState
  }

  /// If the last index is not reached yet, produces the next output tuple and
  /// increments the current index in the state; returns an empty optional
  /// otherwise.
  func private @iteratorsSampleInputOperatorNext(%state: !sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState) {
    %currentIndex = llvm.extractvalue %state[0] : !sampleInputOperatorState
    %lastIndex = llvm.extractvalue %state[1] : !sampleInputOperatorState
    %hasNext = arith.cmpi "slt", %currentIndex, %lastIndex : i32

    %maybeTuple, %updatedState = scf.if %hasNext -> (!optionalIntTuple, !sampleInputOperatorState) {
      // Update state (i.e., increment current index)
      %one = arith.constant 1 : i32
      %nextIndex = arith.addi %currentIndex, %one : i32
      %updatedState = llvm.insertvalue %nextIndex, %state[0] : !sampleInputOperatorState

      // Make optional tuple from value at current index.
      %tuple = call @makeTupleFromValue(%nextIndex) : (i32) -> !intTuple
      %ret = call @makeOptionalFromTuple(%tuple) : (!intTuple) -> !optionalIntTuple

      scf.yield %ret, %updatedState : !optionalIntTuple, !sampleInputOperatorState
    } else {
      // Done iterating over sample data --> return empty optional.
      %ret = call @makeEmptyOptional() : () -> !optionalIntTuple
      scf.yield %ret, %state : !optionalIntTuple, !sampleInputOperatorState
    }

    return %maybeTuple, %updatedState : !optionalIntTuple, !sampleInputOperatorState
  }

  /// "Closes" the iterator (no-op in this case).
  func private @iteratorsSampleInputOperatorClose(%state: !sampleInputOperatorState) -> !sampleInputOperatorState {
    return %state : !sampleInputOperatorState
  }

  //===--------------------------------------------------------------------===//
  // Reduce operator
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

  /// Creates an initial state of the reduce operator, which consists of the
  /// state of its upstream operator. Since that could be of any type, the
  /// concrete instatiation of this function should be generated.
  func private @iteratorsReduceOperatorCreate(%upstreamState: !sampleInputOperatorState) -> !reduceOperatorState {
    %emptyState = llvm.mlir.undef : !reduceOperatorState
    %completeState = llvm.insertvalue %upstreamState, %emptyState[0] : !reduceOperatorState
    return %completeState : !reduceOperatorState
  }

  /// "Opens" the reduce operator (which consists of opening its upstream).
  func private @iteratorsReduceOperatorOpen(%state: !reduceOperatorState) -> !reduceOperatorState {
    %upstreamState = llvm.extractvalue %state[0] : !reduceOperatorState
    %updatedUpstreamState = call @iteratorsSampleInputOperatorOpen(%upstreamState) : (!sampleInputOperatorState) -> !sampleInputOperatorState
    %updatedState = llvm.insertvalue %updatedUpstreamState, %state[0] : !reduceOperatorState
    return %updatedState : !reduceOperatorState
  }

  /// Produces the next output tuple. In the first call, this consists of
  /// consuming the entire upstream and combining them pairwise using a
  /// reduce function and returning the result. (We use a placeholder here; the
  /// usage of that function should in fact be generated.) In the next call,
  /// upstream returns an empty optional, so this function does the same.
  func private @iteratorsReduceOperatorNext(%state: !reduceOperatorState) -> (!optionalIntTuple, !reduceOperatorState) {
    %upstreamState = llvm.extractvalue %state[0] : !reduceOperatorState

    // Consume first tuple.
    %maybeFirstTuple, %upstreamStateAfterFirstNext = call @iteratorsSampleInputOperatorNext(%upstreamState) : (!sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState)
    %firstTupleHasValue = llvm.extractvalue %maybeFirstTuple[0] : !optionalIntTuple

    // Run main loop of consuming upstream.
    %retTuple, %consumedUpstreamState = scf.if %firstTupleHasValue -> (!optionalIntTuple, !sampleInputOperatorState) {
      %firstTuple = llvm.extractvalue %maybeFirstTuple[1] : !optionalIntTuple

      %emptyOptional, %agg, %consumedUpstreamState = scf.while (%agg = %firstTuple, %currentUpstreamState = %upstreamStateAfterFirstNext) : (!intTuple, !sampleInputOperatorState) -> (!optionalIntTuple, !intTuple, !sampleInputOperatorState) {
        %maybeTuple, %nextUpstreamState = call @iteratorsSampleInputOperatorNext(%currentUpstreamState) : (!sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState)
        %hasValue = llvm.extractvalue %maybeTuple[0] : !optionalIntTuple
        scf.condition(%hasValue) %maybeTuple, %agg, %nextUpstreamState : !optionalIntTuple, !intTuple, !sampleInputOperatorState
      } do {
      ^bb0(%maybeTuple: !optionalIntTuple, %agg: !intTuple, %nextUpstreamState: !sampleInputOperatorState):
        %tuple =  llvm.extractvalue %maybeTuple[1] : !optionalIntTuple
        %newAgg = call @iteratorsSum(%agg, %tuple) : (!intTuple, !intTuple) -> (!intTuple)
        scf.yield %newAgg, %nextUpstreamState : !intTuple, !sampleInputOperatorState
      }

      // Return aggregate wrapped in optional.
      %retTuple = call @makeOptionalFromTuple(%agg) : (!intTuple) -> !optionalIntTuple
      scf.yield %retTuple, %consumedUpstreamState : !optionalIntTuple, !sampleInputOperatorState
    } else {
      // Upstream provided no tuple --> return empty optional.
      %retTuple = call @makeEmptyOptional() : () -> !optionalIntTuple
      scf.yield %retTuple, %upstreamStateAfterFirstNext : !optionalIntTuple, !sampleInputOperatorState
    }

    // Update upstream state.
    %finalState = llvm.insertvalue %consumedUpstreamState, %state[0] : !reduceOperatorState

    return %retTuple, %finalState : !optionalIntTuple, !reduceOperatorState
  }

  /// "Closes" the reduce operator (which consists of closing its upstream).
  func private @iteratorsReduceOperatorClose(%state: !reduceOperatorState) -> !reduceOperatorState {
    %upstreamState = llvm.extractvalue %state[0] : !reduceOperatorState
    %updatedUpstreamState = call @iteratorsSampleInputOperatorClose(%upstreamState) : (!sampleInputOperatorState) -> (!sampleInputOperatorState)
    %updatedState = llvm.insertvalue %updatedUpstreamState, %state[0] : !reduceOperatorState
    return %updatedState : !reduceOperatorState
  }

  //===--------------------------------------------------------------------===//
  // Pipeline driver (consume and print)
  //===--------------------------------------------------------------------===//

  /// Drives the pipelines, i.e., opens the root operator, consumes all of its
  /// tuples, prints each of them, and closes the operator again.
  func private @iteratorsConsumeAndPrint(%state: !reduceOperatorState) {
    %openedState = call @iteratorsReduceOperatorOpen(%state) : (!reduceOperatorState) -> !reduceOperatorState
    %emptyTuple, %consumedState = scf.while(%currentState = %openedState) : (!reduceOperatorState) -> (!optionalIntTuple, !reduceOperatorState) {
      %maybeTuple, %nextState = call @iteratorsReduceOperatorNext(%currentState) : (!reduceOperatorState) -> (!optionalIntTuple, !reduceOperatorState)
      %hasValue = llvm.extractvalue %maybeTuple[0] : !optionalIntTuple
      scf.condition(%hasValue) %maybeTuple, %nextState : !optionalIntTuple, !reduceOperatorState
    } do {
    ^bb0(%maybeTuple: !optionalIntTuple, %nextState: !reduceOperatorState):
      %tuple = llvm.extractvalue %maybeTuple[1] : !optionalIntTuple
      call @iteratorsPrintTuple(%tuple) : (!intTuple) -> ()
      scf.yield %nextState : !reduceOperatorState
    }
    %finalState = call @iteratorsReduceOperatorClose(%consumedState) : (!reduceOperatorState) -> !reduceOperatorState
    return
  }

  //===--------------------------------------------------------------------===//
  // Tests
  //===--------------------------------------------------------------------===//
  func @main() {
    // Optional and printing -------------------------------------------------//
    %leet = arith.constant 1337 : i32
    %leetTuple = call @makeTupleFromValue(%leet) : (i32) -> !intTuple
    call @iteratorsPrintTuple(%leetTuple) : (!intTuple) -> ()
    // CHECK: (1337)

    %emptyOptional = call @makeEmptyOptional() : () -> !optionalIntTuple
    call @iteratorsPrintOptionalTuple(%emptyOptional) :  (!optionalIntTuple) -> ()
    // CHECK-NEXT: ()

    %fourtyTwo = arith.constant 42 : i32
    %fourtyTwoTuple = call @makeTupleFromValue(%fourtyTwo) : (i32) -> !intTuple
    %fourtyTwoMaybeTuple = call @makeOptionalFromTuple(%fourtyTwoTuple) : (!intTuple) -> !optionalIntTuple
    call @iteratorsPrintOptionalTuple(%fourtyTwoMaybeTuple) :  (!optionalIntTuple) -> ()
    // CHECK-NEXT: (42)

    // Sum -------------------------------------------------------------------//

    %five = arith.constant 5 : i32
    %fiveTuple = call @makeTupleFromValue(%five) : (i32) -> !intTuple
    %resultTuple = call @iteratorsSum(%fiveTuple, %fiveTuple) : (!intTuple, !intTuple) -> !intTuple
    call @iteratorsPrintTuple(%resultTuple) : (!intTuple) -> ()
    // CHECK-NEXT: (10)

    // Sample input operator -------------------------------------------------//

    // Create initial state of input operator.
    %initialInputState = call @iteratorsSampleInputOperatorCreate() : () -> !sampleInputOperatorState

    // Manually drain that operator.
    %openedState = call @iteratorsSampleInputOperatorOpen(%initialInputState) : (!sampleInputOperatorState) -> !sampleInputOperatorState
    %firstMaybeTuple, %firstConsumedState = call @iteratorsSampleInputOperatorNext(%openedState) : (!sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState)
    call @iteratorsPrintOptionalTuple(%firstMaybeTuple) : (!optionalIntTuple) -> ()
    // CHECK-NEXT: (1)
    %secondMaybeTuple, %secondConsumedState = call @iteratorsSampleInputOperatorNext(%firstConsumedState) : (!sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState)
    call @iteratorsPrintOptionalTuple(%secondMaybeTuple) : (!optionalIntTuple) -> ()
    // CHECK-NEXT: (2)
    %thirdMaybeTuple, %thirdConsumedState = call @iteratorsSampleInputOperatorNext(%secondConsumedState) : (!sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState)
    call @iteratorsPrintOptionalTuple(%thirdMaybeTuple) : (!optionalIntTuple) -> ()
    // CHECK-NEXT: (3)
    %fourthMaybeTuple, %fourthConsumedState = call @iteratorsSampleInputOperatorNext(%thirdConsumedState) : (!sampleInputOperatorState) -> (!optionalIntTuple, !sampleInputOperatorState)
    call @iteratorsPrintOptionalTuple(%fourthMaybeTuple) : (!optionalIntTuple) -> ()
    // CHECK-NEXT: ()

    // Reduce operator -------------------------------------------------------//

    // Create initial state based on previous operator state.
    %initialReduceState = call @iteratorsReduceOperatorCreate(%initialInputState) : (!sampleInputOperatorState) -> !reduceOperatorState

    // Manually drain that operator.
    %openedReduceState = call @iteratorsReduceOperatorOpen(%initialReduceState) : (!reduceOperatorState) -> !reduceOperatorState
    %firstMaybeReduceTuple, %firstConsumedReduceState = call @iteratorsReduceOperatorNext(%openedReduceState) : (!reduceOperatorState) -> (!optionalIntTuple, !reduceOperatorState)
    call @iteratorsPrintOptionalTuple(%firstMaybeReduceTuple) : (!optionalIntTuple) -> ()
    // CHECK-NEXT: (6)
    %secondMaybeReduceTuple, %secondConsumedReduceState = call @iteratorsReduceOperatorNext(%firstConsumedReduceState) : (!reduceOperatorState) -> (!optionalIntTuple, !reduceOperatorState)
    call @iteratorsPrintOptionalTuple(%secondMaybeReduceTuple) : (!optionalIntTuple) -> ()
    // CHECK-NEXT: ()

    // Print and consume -----------------------------------------------------//
    call @iteratorsConsumeAndPrint(%initialReduceState) : (!reduceOperatorState) -> ()
    // CHECK: (6)

    return
  }
}
