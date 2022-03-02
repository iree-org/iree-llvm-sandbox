// RUN: mlir-proto-opt -linalg-interp-transforms %s | FileCheck %s

// This test is verifying that a non-trivial 2*tiling+padding+vectorization transformation completes successfully

// CHECK-LABEL: func @matmul_tensors(
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}

pdl.pattern @pdl_target: benefit(1) {
  %args = operands
  %results= types
  %0 = operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  rewrite %0 with "linalg_transform.apply"
}
linalg_transform.sequence {
  %0 = match @pdl_target
  %1 = tile %0 {generalize = false, hoist_paddings = [], interchange = [0, 2, 1], pack_paddings = [], pad = false, peel = [], scalarize_dyn_dims = false, sizes = [32, 32, 32], transpose_paddings = []}
  %2 = tile %1 {generalize = false, hoist_paddings = [], interchange = [0, 1, 2], pack_paddings = [], pad = false, peel = [], scalarize_dyn_dims = false, sizes = [4, 4, 1], transpose_paddings = []}
  %3 = tile %2 {generalize = false, hoist_paddings = [4, 3, 0], interchange = [], pack_paddings = [1, 1, 0], pad = true, peel = [], scalarize_dyn_dims = false, sizes = [], transpose_paddings = [[1, 0], [0, 1]]}
  %4 = vectorize %3  {vectorize_padding = true}
}
