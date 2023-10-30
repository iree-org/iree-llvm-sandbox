// RUN: jasc-opt %s -jasc-apply-transform-script='enforce-single-top-level-transform-op=0' \
// RUN:             -jasc-erase-transform-script \
// RUN:             -jasc-apply-transform-script='enforce-single-top-level-transform-op=0' \
// RUN:   | FileCheck %s

// This test specializes a parametric transform script and then applies it to a
// computation.
// This is accomplished using two transform interpreter passes. The first to
// apply the first transform script to the second. In this process the value of
// the transform.jasc.tuning_param is set according to the config in the first
// script (16). In the second pass this specialized schedule is applied to the
// computation to tile the `linalg.matmul` with tile size 16.

// Meta schedule to specialize the parametric schedule
transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %parametric = transform.structured.match ops{["transform.sequence"]} in %arg0 : (!transform.any_op) -> !transform.op<"transform.sequence">
  transform.jasc.apply_tuning_config %parametric {config = [16 : i32]} : !transform.op<"transform.sequence">
}

// Parametric schedule to tile a matmul
transform.sequence failures(propagate) {
^bb1(%arg0: !transform.any_op):
  %0 = transform.jasc.tuning_param {default_value = 0 : i32} -> !transform.param<i32>
  // CHECK: transform.jasc.tuning_param {default_value = 0 : i32
  // CHECK-SAME: tuned_value = 16 : i32}
  %1 = transform.structured.match ops{["linalg.matmul"]} in %arg0 : (!transform.any_op) -> !transform.op<"linalg.matmul">
  %tiled_linalg_op, %loops = transform.structured.tile_using_for %1[%0] : (!transform.op<"linalg.matmul">, !transform.param<i32>) -> (!transform.op<"linalg.matmul">, !transform.any_op)
}

// CHECK-LABEL: @matmul
func.func private @matmul(%lhs : tensor<64x64xi32>, %rhs : tensor<64x64xi32>) -> tensor<64x64xi32> {
  %c_0 = arith.constant 0 : i32
  %init_acc_uninitialized =  tensor.empty() : tensor<64x64xi32>
  %zero_acc = linalg.fill ins(%c_0 : i32) outs(%init_acc_uninitialized : tensor<64x64xi32>) -> tensor<64x64xi32>
  %matmul_result = linalg.matmul ins(%lhs, %rhs : tensor<64x64xi32>, tensor<64x64xi32>) outs(%zero_acc : tensor<64x64xi32>) -> tensor<64x64xi32>
  // CHECK: scf.for
  return %matmul_result : tensor<64x64xi32>
}
