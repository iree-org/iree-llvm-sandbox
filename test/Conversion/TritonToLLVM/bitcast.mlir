// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<1024xi32>
// CHECK-SAME:      ) -> tensor<1024xf32> {
// CHECK-NEXT:    %[[V0:.*]] = tensor.bitcast %[[ARG0]] : tensor<1024xi32> to tensor<1024xf32>
// CHECK-NEXT:    return %[[V0]] : tensor<1024xf32>
func.func public @kernel(%arg0: tensor<1024xi32>) -> tensor<1024xf32> {
  %0 = tt.bitcast %arg0 : tensor<1024xi32> -> tensor<1024xf32>
  return %0 : tensor<1024xf32>
}
