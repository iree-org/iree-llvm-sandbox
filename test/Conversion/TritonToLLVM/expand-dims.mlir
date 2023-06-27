// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> tensor<1x4x1xi32> {
// CHECK-NEXT:    %[[V0:.*]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<1x4xi32>
// CHECK-NEXT:    %[[V1:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0], [1, 2]] : tensor<1x4xi32> into tensor<1x4x1xi32>
// CHECK-NEXT:    %[[V2:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0], [1, 2]] : tensor<1x4xi32> into tensor<1x1x4xi32>
// CHECK-NEXT:    return %[[V1]] : tensor<1x4x1xi32>
func.func public @kernel(%arg0: tensor<4xi32>) -> tensor<1x4x1xi32> {
  %0 = tt.expand_dims %arg0 {axis = 0 : i32} : (tensor<4xi32>) -> tensor<1x4xi32>
  %1 = tt.expand_dims %0 {axis = 2 : i32} : (tensor<1x4xi32>) -> tensor<1x4x1xi32>
  %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<1x4xi32>) -> tensor<1x1x4xi32>
  return %1 : tensor<1x4x1xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> tensor<1x4x1xi32> {
// CHECK-NEXT:    %[[V0:.*]] = tensor.expand_shape %[[ARG0]] {{\[}}[0, 1]] : tensor<4xi32> into tensor<4x1xi32>
// CHECK-NEXT:    %[[V1:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0, 1], [2]] : tensor<4x1xi32> into tensor<1x4x1xi32>
// CHECK-NEXT:    %[[V2:.*]] = tensor.expand_shape %[[V0]] {{\[}}[0], [1, 2]] : tensor<4x1xi32> into tensor<4x1x1xi32>
// CHECK-NEXT:    return %[[V1]] :  tensor<1x4x1xi32>
func.func public @kernel(%arg0: tensor<4xi32>) -> tensor<1x4x1xi32> {
  %0 = tt.expand_dims %arg0 {axis = 1 : i32} : (tensor<4xi32>) -> tensor<4x1xi32>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : (tensor<4x1xi32>) -> tensor<1x4x1xi32>
  %2 = tt.expand_dims %0 {axis = 1 : i32} : (tensor<4x1xi32>) -> tensor<4x1x1xi32>
  return %1 : tensor<1x4x1xi32>
}
