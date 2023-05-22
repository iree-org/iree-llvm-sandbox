// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: f64)
// CHECK-NEXT:    %[[V0:.*]] = tensor.splat %[[ARG0]] : tensor<4xf64>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: f64) {
  %0 = tt.splat %arg0 : (f64) -> tensor<4xf64>
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f64, 1>)
// CHECK-NEXT:    %[[V0:.*]] = llvm.ptrtoint %[[ARG0]] : !llvm.ptr<f64, 1> to i64
// CHECK-NEXT:    %[[V1:.*]] = arith.index_cast %[[V0]] : i64 to index
// CHECK-NEXT:    %[[V2:.*]] = tensor.splat %[[V1]] : tensor<4xindex>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<f64>) {
  %0 = tt.splat %arg0 : (!tt.ptr<f64>) -> tensor<4x!tt.ptr<f64>>
  return
}
