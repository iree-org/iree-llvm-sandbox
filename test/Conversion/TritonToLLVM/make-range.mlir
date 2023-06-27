// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-NEXT:    %[[V0:.*]] = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
// CHECK-NEXT:    return %[[V0]] : tensor<4xi32>
func.func public @kernel() -> tensor<4xi32> {
  %0 = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  return %0 : tensor<4xi32>
}
