// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt --verify-diagnostics --split-input-file %s

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>
func private @makeSampleInputState() -> !sampleInputState
func @testOpenElementTypeMismatch() {
  %initialState = call @makeSampleInputState() : () -> !sampleInputState
  // expected-error@+1 {{Opening iterator of type '!iterators.sampleinputstate<i32>' should return the same type but returns '!iterators.sampleinputstate<i64>'}}
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !iterators.sampleinputstate<i64>
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>
func private @makeSampleInputState() -> !sampleInputState
func @testOpenStateTypeMismatch() {
  %initialState = call @makeSampleInputState() : () -> !sampleInputState
  // expected-error@+1 {{Opening iterator of type '!iterators.sampleinputstate<i32>' should return the same type but returns '!iterators.reducestate<!iterators.sampleinputstate<i32>>'}}
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !reduceState
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>
func private @makeSampleInputState() -> !sampleInputState
func @testReduceStateTypeMismatch() {
  %initialState = call @makeSampleInputState() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  // expected-error@+1 {{Type mismatch: Consuming an element of an iterator of type '!iterators.sampleinputstate<i32>' should return in an iterator of the same type but returns '!iterators.sampleinputstate<i64>'}}
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!sampleInputState) -> (!iterators.sampleinputstate<i64>, i1, i32)
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>
func private @makeSampleInputState() -> !sampleInputState
func @testReduceElementTypeMismatch() {
  %initialState = call @makeSampleInputState() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  // expected-error@+1 {{Type mismatch: Element returned by iterator of type '!iterators.sampleinputstate<i32>' should be 'i32' but is 'i64'}}
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!sampleInputState) -> (!sampleInputState, i1, i64)
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<i32>
!reduceState = type !iterators.reducestate<!sampleInputState>
func private @makeSampleInputState() -> !sampleInputState
func @testCloseStateTypeMismatch() {
  %initialState = call @makeSampleInputState() : () -> !sampleInputState
  %openedState = "iterators.open"(%initialState) : (!sampleInputState) -> !sampleInputState
  // expected-error@+1 {{Type mismatch: Closing iterator of type '!iterators.sampleinputstate<i32>' should return the same type but returns '!iterators.sampleinputstate<i64>'}}
  %closedState = "iterators.close"(%openedState) : (!sampleInputState) -> !iterators.sampleinputstate<i64>
  return
}
