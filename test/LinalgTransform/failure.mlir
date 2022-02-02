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

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func public @benchmark(%arg0: tensor<39x154xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<154x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<39x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> tensor<39x5xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<39x154xf32>, tensor<154x5xf32>) outs(%arg2 : tensor<39x5xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %1 = arith.mulf %arg3, %arg4 : f32
    %2 = arith.addf %arg5, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<39x5xf32>
  return %0 : tensor<39x5xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  apply_native_constraint "nestedInFunc" [@benchmark](%2 : !pdl.operation)
  rewrite %2 with "linalg_transform.apply"
}

linalg_transform.sequence {
  %0 = tile when @target_pattern {interchange = [0, 2, 1],  sizes = [3, 5, 14]}
  %1 = tile %0 { sizes = [3, 5, 2]}
  // expected-error@below {{failed to apply}}
  %2 = vectorize %1 {vectorize_padding = true}
}


// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

func public @benchmark(%arg0: tensor<39x154xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<154x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<39x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> tensor<39x5xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
  %0 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<39x154xf32>, tensor<154x5xf32>) outs(%arg2 : tensor<39x5xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
    %1 = arith.mulf %arg3, %arg4 : f32
    %2 = arith.addf %arg5, %1 : f32
    linalg.yield %2 : f32
  } -> tensor<39x5xf32>
  return %0 : tensor<39x5xf32>
}

pdl.pattern @target_pattern : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  apply_native_constraint "nestedInFunc" [@benchmark](%2 : !pdl.operation)
  rewrite %2 with "linalg_transform.apply"
}

linalg_transform.sequence {
  %0 = tile when @target_pattern {interchange = [2, 1, 0],  sizes = [3, 5, 14]}
  %1 = tile %0 { sizes = [3, 5, 2]}
  // expected-error@below {{failed to apply}}
  %2 = vectorize %1 {vectorize_padding = true}
}
