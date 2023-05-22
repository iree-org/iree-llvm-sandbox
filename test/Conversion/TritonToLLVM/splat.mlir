// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm \
// RUN: | FileCheck %s

// CHECK-LABEL: llvm.func @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: f64)
// CHECK-NEXT:    %[[V0:.*]] = tensor.splat %[[ARG0]] : tensor<4xf64>
// CHECK-NEXT:    llvm.return
tt.func public @kernel(%arg0: f64) {
  %0 = tt.splat %arg0 : (f64) -> tensor<4xf64>
  tt.return
}
