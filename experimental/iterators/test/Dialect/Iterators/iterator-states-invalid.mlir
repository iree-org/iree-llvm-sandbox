// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt --verify-diagnostics --split-input-file %s

func private @makeReduceState() -> !iterators.reducestate<i32>
func @testOpenElementTypeMismatch() {
  %initialState = call @makeReduceState() : () -> !iterators.reducestate<i32>
  // expected-error@+1 {{Opening iterator of type '!iterators.reducestate<i32>' should return the same type but returns '!iterators.reducestate<i64>'}}
  %openedState = "iterators.open"(%initialState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i64>
  return
}

// -----

func private @makeReduceState() -> !iterators.reducestate<i32>
func @testOpenStateTypeMismatch() {
  %initialState = call @makeReduceState() : () -> !iterators.reducestate<i32>
  // expected-error@+1 {{Opening iterator of type '!iterators.reducestate<i32>' should return the same type but returns '!iterators.sampleinputstate<i32>'}}
  %openedState = "iterators.open"(%initialState) : (!iterators.reducestate<i32>) -> !iterators.sampleinputstate<i32>
  return
}

// -----

func private @makeReduceState() -> !iterators.reducestate<i32>
func @testReduceStateTypeMismatch() {
  %initialState = call @makeReduceState() : () -> !iterators.reducestate<i32>
  %openedState = "iterators.open"(%initialState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i32>
  // expected-error@+1 {{Type mismatch: Consuming an element of an iterator of type '!iterators.reducestate<i32>' should return in an iterator of the same type but returns '!iterators.reducestate<i64>'}}
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!iterators.reducestate<i32>) -> (!iterators.reducestate<i64>, i1, i32)
  return
}

// -----

func private @makeReduceState() -> !iterators.reducestate<i32>
func @testReduceElementTypeMismatch() {
  %initialState = call @makeReduceState() : () -> !iterators.reducestate<i32>
  %openedState = "iterators.open"(%initialState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i32>
  // expected-error@+1 {{Type mismatch: Element returned by iterator of type '!iterators.reducestate<i32>' should be 'i32' but is 'i64'}}
  %consumedState, %hasNext, %nextElement = "iterators.next"(%openedState) : (!iterators.reducestate<i32>) -> (!iterators.reducestate<i32>, i1, i64)
  return
}

// -----

func private @makeReduceState() -> !iterators.reducestate<i32>
func @testCloseStateTypeMismatch() {
  %initialState = call @makeReduceState() : () -> !iterators.reducestate<i32>
  %openedState = "iterators.open"(%initialState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i32>
  // expected-error@+1 {{Type mismatch: Closing iterator of type '!iterators.reducestate<i32>' should return the same type but returns '!iterators.reducestate<i64>'}}
  %closedState = "iterators.close"(%openedState) : (!iterators.reducestate<i32>) -> !iterators.reducestate<i64>
  return
}
