// Test that we can parse and verify iterator states without errors
// RUN: mlir-proto-opt --verify-diagnostics --split-input-file %s

!sampleInputState = type !iterators.sampleinputstate<tuple<i32>>
func @testCreateSampleInputTypeMismatch() {
  // expected-error@+1 {{Sample input iterator (currently) has to return elements of type 'tuple<i32>'}}
  %initialState = "iterators.createSampleInputState"() : () -> !iterators.sampleinputstate<tuple<i64>>
  return
}

// -----

!sampleInputState = type !iterators.sampleinputstate<tuple<i32>>
!wrongSampleInputState = type !iterators.sampleinputstate<tuple<i64>>
!wrongReduceState = type !iterators.reducestate<!wrongSampleInputState>
func @testCreateSampleInputTypeMismatch() {
  %initialUpstreamState = "iterators.createSampleInputState"() : () -> !sampleInputState
  // expected-error@+1 {{Upstream iterator of reduce iterator must produce elements of type 'tuple<i64>' but produces elements of type 'tuple<i32>'}}
  %initialState = "iterators.createReduceState"(%initialUpstreamState) : (!sampleInputState) -> !wrongReduceState
  return
}
