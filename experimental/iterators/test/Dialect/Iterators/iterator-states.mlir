// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt %s

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>

func private @makeReduceState() -> !reduceState
func private @makeSampleInputState() -> !sampleInputState

func @testSampleInput() {
  %initialState = call @makeSampleInputState() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!sampleInputState) -> (!sampleInputState, i1, i32)
  %closedState = "iterators.close"(%consumedState) : (!sampleInputState) -> !sampleInputState
  return
}

func @testReduce() {
  %initialState = call @makeReduceState() : () -> !reduceState
  %openedState = "iterators.open"(%initialState) : (!reduceState) -> !reduceState
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!reduceState) -> (!reduceState, i1, i32)
  %closedState = "iterators.close"(%consumedState) : (!reduceState) -> !reduceState
  return
}
