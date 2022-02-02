// RUN: mlir-proto-opt -linalg-interp-transforms -split-input-file -verify-diagnostics

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
  // expected-error@below {{failed to apply}}
  vectorize when @target_pattern
}

