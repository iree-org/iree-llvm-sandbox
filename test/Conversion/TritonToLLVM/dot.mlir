// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel()
// CHECK-DAG:     %[[V0:.*]] = arith.constant dense<{{.*}}> : tensor<2x4xf32>
// CHECK-DAG:     %[[V1:.*]] = arith.constant dense<{{.*}}> : tensor<4x8xf32>
// CHECK-DAG:     %[[V2:.*]] = arith.constant dense<{{.*}}> : tensor<2x8xf32>
// CHECK-DAG:     %[[V3:.*]] = linalg.matmul ins(%[[V0]], %[[V1]] : tensor<2x4xf32>, tensor<4x8xf32>) outs(%[[V2]] : tensor<2x8xf32>) -> tensor<2x8xf32>
// CHECK-NEXT:    return
func.func public @kernel() {
  %0 = arith.constant dense<0.> : tensor<2x4xf32>
  %1 = arith.constant dense<0.> : tensor<4x8xf32>
  %2 = arith.constant dense<0.> : tensor<2x8xf32>
  %3 = tt.dot %0, %1, %2 {allowTF32 = true} : tensor<2x4xf32> * tensor<4x8xf32> -> tensor<2x8xf32>
  return
}
