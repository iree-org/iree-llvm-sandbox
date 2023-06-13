// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<i32, 1>, i32) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<i32>, i32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>)
// CHECK-DAG:     %[[V0:.*]] = arith.constant dense<0> : tensor<1xi32>
// CHECK-DAG:     %[[V1:.*]] = tensor.splat %{{.*}} : tensor<1xindex>
// CHECK-DAG:     %[[V2:.*]] = arith.index_cast %[[V0]] : tensor<1xi32> to tensor<1xindex>
// CHECK-DAG:     %[[V3:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[V4:.*]] = tensor.splat %[[V3]] : tensor<1xindex>
// CHECK-DAG:     %[[V5:.*]] = arith.muli %[[V2]], %[[V4]] : tensor<1xindex>
// CHECK-DAG:     %[[V6:.*]] = arith.addi %[[V1]], %[[V5]] : tensor<1xindex>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>) {
  %0 = arith.constant dense<[0]> : tensor<1xi32>
  %1 = tt.splat %arg0 : (!tt.ptr<i32>) -> tensor<1x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
  return
}
