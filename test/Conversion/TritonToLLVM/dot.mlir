// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x4xf32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4x8xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x8xf32>) -> tensor<2x8xf32> {
// CHECK-DAG:     %[[V0:.*]] = linalg.matmul ins(%[[ARG0]], %[[ARG1]] : tensor<2x4xf32>, tensor<4x8xf32>) outs(%[[ARG2]] : tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:    return %[[V0]] : tensor<2x8xf32>
func.func public @kernel(%arg0: tensor<2x4xf32>, %arg1: tensor<4x8xf32>, %arg2: tensor<2x8xf32>) -> tensor<2x8xf32> {
  %0 = tt.dot %arg0, %arg1, %arg2 {allowTF32 = true} : tensor<2x4xf32> * tensor<4x8xf32> -> tensor<2x8xf32>
  return %0 : tensor<2x8xf32>
}
