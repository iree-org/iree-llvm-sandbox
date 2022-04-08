// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt %s

func private @makeReduceState() -> !iterators.reducestate<i32>
func private @makeSampleInputState() -> !iterators.sampleinputstate<i32>

func @testSampleInput() {
  %initialState = call @makeSampleInputState() : () -> !iterators.sampleinputstate<i32>
  %openedState = "iterators.open"(%initialState) : (!iterators.sampleinputstate<i32>) -> !iterators.sampleinputstate<i32>
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!iterators.sampleinputstate<i32>) -> (!iterators.sampleinputstate<i32>, i1, i32)
  %closedState = "iterators.close"(%consumedState) : (!iterators.sampleinputstate<i32>) -> !iterators.sampleinputstate<i32>
  return
}

func @testReduce() {
  %initialState = call @makeReduceState() : () -> !iterators.reducestate<i32>
  %openedState = "iterators.open"(%initialState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i32>
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!iterators.reducestate<i32>) -> (!iterators.reducestate<i32>, i1, i32)
  %closedState = "iterators.close"(%consumedState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i32>
  return
}
