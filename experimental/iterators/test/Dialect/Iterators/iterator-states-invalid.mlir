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
