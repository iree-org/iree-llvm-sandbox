// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> tensor<2x2xi32> {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant dense<2> : tensor<2xindex>
// CHECK-NEXT:    %[[V1:.*]] = tensor.reshape %[[ARG0]](%[[V0]]) : (tensor<4xi32>, tensor<2xindex>) -> tensor<2x2xi32>
// CHECK-NEXT:    return %[[V1]] : tensor<2x2xi32>
func.func public @kernel(%arg0: tensor<4xi32>) -> tensor<2x2xi32> {
  %0 = tt.view %arg0 : (tensor<4xi32>) -> tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}
