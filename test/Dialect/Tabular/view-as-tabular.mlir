// RUN: structured-opt %s \
// RUN: | FileCheck %s

func.func @main() {
  // CHECK-LABEL: func.func @main() {
  %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  // CHECK-NEXT:    %[[V0:.*]] = arith.constant
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  // CHECK-NEXT:    %[[V1:.*]] = bufferization.to_memref
  %view = tabular.view_as_tabular %m1
    : (memref<3xi32>) -> !tabular.tabular_view<i32>
  // CHECK-NEXT:    %[[V2:tabularview.*]] = tabular.view_as_tabular %[[V1]] : (memref<3xi32>) -> !tabular.tabular_view<i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
