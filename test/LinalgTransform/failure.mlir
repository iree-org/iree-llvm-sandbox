// RUN: mlir-proto-opt -linalg-interp-transforms -split-input-file -verify-diagnostics -allow-unregistered-dialect

// This cannot be vectorized because of dynamic tensor shapes. We expect the
// pass fail and report an error at the vectorization operation below.
func public @non_vectorizable(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%arg0: tensor<?xf32>) outs(%arg1: tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = arith.mulf %arg2, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  rewrite %2 with "linalg_transform.apply"
}

linalg_transform.sequence {
  %0 = match @target_pattern
  // expected-error@below {{failed to apply}}
  vectorize %0
}

// -----

func public @no_loop(%arg0: tensor<?xf32>, %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %0 = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]}
    ins(%arg0: tensor<?xf32>) outs(%arg1: tensor<?xf32>) {
  ^bb0(%arg2: f32, %arg3: f32):
    %1 = arith.mulf %arg2, %arg2 : f32
    linalg.yield %1 : f32
  } -> tensor<?xf32>
  return %0 : tensor<?xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  rewrite %2 with "linalg_transform.apply"
}

linalg_transform.sequence {
  %0 = match @target_pattern
  // expected-error@below {{the transformed op is enclosed by 0 loops, but 1 expected}}
  // expected-error@below {{failed to apply}}
  get_parent_loop %0
}

// -----

func private @prevent_dce()

pdl.pattern @something : benefit(1) {
  %0 = operands
  %2 = operation "scf.for"(%0 : !pdl.range<value>)
  rewrite %2 with "linalg_transform.apply"
}

func public @loop(%lb: index, %ub: index, %step: index) {
  scf.for %i = %lb to %ub step %step {
    call @prevent_dce() : () -> ()
  }
  return
}

linalg_transform.sequence {
  %0 = match @something
  // expected-error@below {{NYI: cannot target the result of pipelining}}
  // expected-error@below {{failed to apply}}
  %1 = pipeline_loop %0
  // expected-note@below {{use here}}
  get_parent_loop %1
}

// -----

func public @no_outlining() {
  "some.operation"() ({}, {}) : () -> ()
  return
}

pdl.pattern @some_operation : benefit(1) {
  %0 = operation "some.operation"
  rewrite %0 with "linalg_transform.apply"
}

linalg_transform.sequence {
  %0 = match @some_operation
  // Make sure we don't crash on wrong operation type.
  // expected-error@below {{failed to apply}}
  outline_loop %0 {func_name = "outlined"}
}
