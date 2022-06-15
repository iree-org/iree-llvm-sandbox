// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt %s

!sampleInputState = !iterators.sampleinputstate<tuple<i32>>
!reduceState = !iterators.reducestate<!sampleInputState>

func.func @testSampleInput() {
  %initialState = "iterators.createSampleInputState"() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!sampleInputState) -> (!sampleInputState, i1, tuple<i32>)
  %closedState = "iterators.close"(%consumedState) : (!sampleInputState) -> !sampleInputState
  return
}

func.func @testReduce() {
  %initialUpstreamState = "iterators.createSampleInputState"() : () -> !sampleInputState
  %initialState = "iterators.createReduceState"(%initialUpstreamState) : (!sampleInputState) -> !reduceState
  %openedState = "iterators.open"(%initialState) : (!reduceState) -> !reduceState
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!reduceState) -> (!reduceState, i1, tuple<i32>)
  %closedState = "iterators.close"(%consumedState) : (!reduceState) -> !reduceState
  return
}
