// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<4xi32>) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[V1:.*]] = tensor.splat %[[V0]] : tensor<i32>
// CHECK-NEXT:    %[[V2:.*]] = linalg.reduce ins(%[[ARG0]] : tensor<4xi32>) outs(%[[V1]] : tensor<i32>) dimensions = [0]
// CHECK-DAG:       (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) {
// CHECK-DAG:         %[[V3:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
// CHECK-DAG:         linalg.yield %[[V3]] : i32
// CHECK-NEXT:      }
// CHECK-DAG:     %[[V4:.*]] = tensor.extract %[[V2]][] : tensor<i32>
// CHECK-DAG:     return %[[V4]] : i32
func.func public @kernel(%arg0: tensor<4xi32>) -> i32 {
  %1 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %2 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %2 : i32
  }) : (tensor<4xi32>) -> i32
  return %1 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x2xi32>) -> tensor<2xi32> {
// CHECK-DAG:     %[[V0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[V1:.*]] = tensor.splat %[[V0]] : tensor<2xi32>
// CHECK-NEXT:    %[[V2:.*]] = linalg.reduce ins(%[[ARG0]] : tensor<2x2xi32>) outs(%[[V1]] : tensor<2xi32>) dimensions = [0]
// CHECK-DAG:       (%[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32) {
// CHECK-DAG:         %[[V3:.*]] = arith.addi %[[ARG1]], %[[ARG2]] : i32
// CHECK-DAG:         linalg.yield %[[V3]] : i32
// CHECK-NEXT:      }
// CHECK-DAG:     return %[[V2]] : tensor<2xi32>
func.func public @kernel(%arg0: tensor<2x2xi32>) -> tensor<2xi32> {
  %1 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
  ^bb0(%arg1: i32, %arg2: i32):
    %2 = arith.addi %arg1, %arg2 : i32
    tt.reduce.return %2 : i32
  }) : (tensor<2x2xi32>) -> tensor<2xi32>
  return %1 : tensor<2xi32>
}


// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x2xi32>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<2x2xf32>
// CHECK-SAME:    ) -> (tensor<2xi32>, tensor<2xf32>) {
// CHECK-DAG:     %[[V0:.*]] = arith.constant 0 : i32
// CHECK-DAG:     %[[V1:.*]] = tensor.splat %[[V0]] : tensor<2xi32>
// CHECK-DAG:     %[[V2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %[[V3:.*]] = tensor.splat %[[V2]] : tensor<2xf32>
// CHECK-NEXT:    %[[V4:.*]]:2 = linalg.reduce ins(%[[ARG0]], %[[ARG1]] : tensor<2x2xi32>, tensor<2x2xf32>) outs(%[[V1]], %[[V3]] : tensor<2xi32>, tensor<2xf32>) dimensions = [0]
// CHECK-DAG:       (%[[ARG2:.*]]: i32, %[[ARG3:.*]]: f32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: f32) {
// CHECK-DAG:         %[[V0:.*]] = arith.addi %[[ARG2]], %[[ARG4]] : i32
// CHECK-DAG:         %[[V1:.*]] = arith.addf %[[ARG3]], %[[ARG5]] : f32
// CHECK-DAG:         linalg.yield %[[V0]], %[[V1]] : i32, f32
// CHECK-NEXT:      }
// CHECK-DAG:     return %[[V4]]#0, %[[V4]]#1 : tensor<2xi32>, tensor<2xf32>
func.func public @kernel(%arg0: tensor<2x2xi32>, %arg1: tensor<2x2xf32>) -> (tensor<2xi32>, tensor<2xf32>) {
  %1:2 = "tt.reduce"(%arg0, %arg1) <{axis = 0 : i32}> ({
  ^bb0(%arg2: i32, %arg3: f32, %arg4: i32, %arg5: f32):
    %2 = arith.addi %arg2, %arg4 : i32
    %3 = arith.addf %arg3, %arg5 : f32
    tt.reduce.return %2, %3 : i32, f32
  }) : (tensor<2x2xi32>, tensor<2x2xf32>) -> (tensor<2xi32>, tensor<2xf32>)
  return %1#0, %1#1 : tensor<2xi32>, tensor<2xf32>
}
