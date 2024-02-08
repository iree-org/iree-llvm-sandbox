// RUN: jasc-opt %s -jasc-apply-transform-script \
// RUN:   | FileCheck %s

transform.sequence failures(suppress) {
^bb0(%arg0: !transform.any_op):
  %linalg_ops = transform.structured.match interface {LinalgOp} in %arg0
    : (!transform.any_op) -> !transform.any_op
  transform.jasc.wrap_in_gpu_launch %linalg_ops
    : (!transform.any_op) -> !transform.op<"gpu.launch">
}

// CHECK-LABEL: func.func @already_wrapped_op
// CHECK-SAME:    %[[ARG0:.*]]: f32,
// CHECK-SAME:    %[[ARG1:.*]]: memref<16xf32>) -> memref<16xf32> {
// CHECK:         gpu.launch
// CHECK-NEXT:      linalg.fill
// CHECK-NEXT:      gpu.terminator
// CHECK-NOT:     gpu
func.func @already_wrapped_op(%arg0: f32, %arg1: memref<16xf32>) -> memref<16xf32> {
  %c1 = arith.constant 1 : index
  gpu.launch blocks(%arg2, %arg3, %arg4)
      in (%arg8 = %c1, %arg9 = %c1, %arg10 = %c1)
      threads(%arg5, %arg6, %arg7)
      in (%arg11 = %c1, %arg12 = %c1, %arg13 = %c1) {
    linalg.fill ins(%arg0 : f32) outs(%arg1 : memref<16xf32>)
    gpu.terminator
  }
  return %arg1 : memref<16xf32>
}

// CHECK-LABEL: func.func @simple_fill
// CHECK-SAME:    %[[ARG0:.*]]: f32,
// CHECK-SAME:    %[[ARG1:.*]]: memref<16xf32>) -> memref<16xf32> {
// CHECK:         gpu.launch
// CHECK-NEXT:      linalg.fill
// CHECK-NEXT:      gpu.terminator
func.func @simple_fill(%arg0: f32, %arg1: memref<16xf32>) -> memref<16xf32> {
  linalg.fill ins(%arg0 : f32) outs(%arg1 : memref<16xf32>)
  return %arg1 : memref<16xf32>
}
