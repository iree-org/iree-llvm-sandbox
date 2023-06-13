// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel()
// CHECK-NEXT:    %[[V2:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-NEXT:    %[[V1:.*]] = arith.constant dense<2> : tensor<2xindex>
// CHECK-NEXT:    %[[V3:.*]] = tensor.reshape %[[V2]](%[[V1]]) : (tensor<4xi32>, tensor<2xindex>) -> tensor<2x2xi32>
// CHECK-NEXT:    return
func.func public @kernel() {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = tt.view %0 : (tensor<4xi32>) -> tensor<2x2xi32>
  return
}
