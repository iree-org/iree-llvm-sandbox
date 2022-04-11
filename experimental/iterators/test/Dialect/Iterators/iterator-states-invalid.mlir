// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt --verify-diagnostics --split-input-file %s

!sampleInputState = type !iterators.sampleinputstate<i32>
func @testCreateSampleInputTypeMismatch() {
  // expected-error@+1 {{Sample input iterator (currently) has to return elements of type 'i32'}}
  %initialState = "iterators.createSampleInputState"() : () -> !iterators.sampleinputstate<i64>
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
!wrongSampleInputState = type !iterators.sampleinputstate<i64>
!wrongReduceState = type !iterators.reducestate<!wrongSampleInputState>
func @testCreateSampleInputTypeMismatch() {
  %initialUpstreamState = "iterators.createSampleInputState"() : () -> !sampleInputState
  // expected-error@+1 {{Upstream iterator of reduce iterator must produce elements of type 'i64' but produces elements of type 'i32'}}
  %initialState = "iterators.createReduceState"(%initialUpstreamState) : (!sampleInputState) -> !wrongReduceState
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
func @testOpenElementTypeMismatch() {
  %initialState = "iterators.createSampleInputState"() : () -> !sampleInputState
  // expected-error@+1 {{'iterators.open' op requires the same type for all operands and results}}
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !iterators.sampleinputstate<i64>
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>
func @testOpenStateTypeMismatch() {
  %initialState = "iterators.createSampleInputState"() : () -> !sampleInputState
  // expected-error@+1 {{'iterators.open' op requires the same type for all operands and results}}
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !reduceState
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
func @testNextStateTypeMismatch() {
  %initialState = "iterators.createSampleInputState"() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  // expected-error@+1 {{'iterators.next' op failed to verify that all of {inputState, resultState} have same type}}
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!sampleInputState) -> (!iterators.sampleinputstate<i64>, i1, i32)
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
func @testNextElementTypeMismatch() {
  %initialState = "iterators.createSampleInputState"() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  // expected-error@+1 {{'iterators.next' op failed to verify that returned element matches element type of state}}
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!sampleInputState) -> (!sampleInputState, i1, i64)
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
func @testCloseStateTypeMismatch() {
  %initialState = "iterators.createSampleInputState"() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  // expected-error@+1 {{'iterators.close' op requires the same type for all operands and results}}
  %closedState = "iterators.close"(%openedState) : (!sampleInputState) -> !iterators.sampleinputstate<i64>
  return
}
