// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32) ->
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<i32, 1>, i32) -> !llvm.ptr<i32, 1>
// CHECK-NEXT:    return %[[V0]] : !llvm.ptr<i32, 1>
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) -> !tt.ptr<i32>{
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<i32>, i32
  return %0 : !tt.ptr<i32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> !llvm.ptr<f32, 1>
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<f32, 1>, i32) -> !llvm.ptr<f32, 1>
// CHECK-NEXT:    return %[[V0]] : !llvm.ptr<f32, 1>
func.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: i32) -> !tt.ptr<f32> {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<f32>, i32
  return %0 : !tt.ptr<f32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i64, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> !llvm.ptr<i64, 1>
// CHECK-NEXT:    %[[V0:.*]] = llvm.getelementptr %[[ARG0]][%[[ARG1]]] : (!llvm.ptr<i64, 1>, i32) -> !llvm.ptr<i64, 1>
// CHECK-NEXT:    return %[[V0]] : !llvm.ptr<i64, 1>
func.func public @kernel(%arg0: !tt.ptr<index>, %arg1: i32) -> !tt.ptr<index> {
  %0 = tt.addptr %arg0, %arg1 : !tt.ptr<index>, i32
  return %0 : !tt.ptr<index>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4xindex>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xi32>) -> tensor<4xindex> {
// CHECK-DAG:     %[[V0:.*]] = arith.index_cast %[[ARG1]] : tensor<4xi32> to tensor<4xindex>
// CHECK-DAG:     %[[V1:.*]] = arith.constant 4 : index
// CHECK-DAG:     %[[V2:.*]] = tensor.splat %[[V1]] : tensor<4xindex>
// CHECK-DAG:     %[[V3:.*]] = arith.muli %[[V0]], %[[V2]] : tensor<4xindex>
// CHECK-DAG:     %[[V4:.*]] = arith.addi %[[ARG0]], %[[V3]] : tensor<4xindex>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: tensor<4x!tt.ptr<i32>>, %arg1: tensor<4xi32>) -> tensor<4x!tt.ptr<i32>> {
  %0 = tt.addptr %arg0, %arg1 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  return %0 : tensor<4x!tt.ptr<i32>>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4xindex>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<4xi32>) -> tensor<4xindex> {
// CHECK-DAG:     %[[V0:.*]] = arith.index_cast %[[ARG1]] : tensor<4xi32> to tensor<4xindex>
// CHECK-DAG:     %[[V1:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[V2:.*]] = tensor.splat %[[V1]] : tensor<4xindex>
// CHECK-DAG:     %[[V3:.*]] = arith.muli %[[V0]], %[[V2]] : tensor<4xindex>
// CHECK-DAG:     %[[V4:.*]] = arith.addi %[[ARG0]], %[[V3]] : tensor<4xindex>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: tensor<4x!tt.ptr<index>>, %arg1: tensor<4xi32>) -> tensor<4x!tt.ptr<index>> {
  %0 = tt.addptr %arg0, %arg1 : tensor<4x!tt.ptr<index>>, tensor<4xi32>
  return %0 : tensor<4x!tt.ptr<index>>
}
