// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = llvm.atomicrmw _and %[[ARG0]], %[[ARG1]] monotonic : !llvm.ptr<i32, 1>, i32
// CHECK-DAG:     return %[[V0]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) -> i32 {
  %0 = "tt.atomic_rmw" (%arg0, %arg1) {atomic_rmw_op = 1 : i32, sem = 1 : i32} : (!tt.ptr<i32>, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<i32, 1>, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1> {
// CHECK-DAG:     %[[V0:.*]] = llvm.atomicrmw xchg %[[ARG0]], %[[ARG1]] monotonic : !llvm.ptr<ptr<i32, 1>, 1>, !llvm.ptr<i32, 1>
// CHECK-DAG:     return %[[V0]] : !llvm.ptr<i32, 1>
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<i32>>, %arg1: !tt.ptr<i32>) -> !tt.ptr<i32> {
  %0 = "tt.atomic_rmw" (%arg0, %arg1) {atomic_rmw_op = 10 : i32, sem = 1 : i32} : (!tt.ptr<!tt.ptr<i32>>, !tt.ptr<i32>) -> !tt.ptr<i32>
  return %0 : !tt.ptr<i32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: f32,
// CHECK-SAME:      %[[ARG2:.*]]: i1) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = scf.if %[[ARG2]] -> (f32) {
// CHECK-DAG:       %[[V1:.*]] = llvm.atomicrmw fadd %[[ARG0]], %[[ARG1]] acquire : !llvm.ptr<f32, 1>, f32
// CHECK-DAG:       scf.yield %[[V1]] : f32
// CHECK-NEXT:    } else {
// CHECK-DAG:       %[[V2:.*]] = llvm.mlir.undef : f32
// CHECK-DAG:       scf.yield %[[V2]] : f32
// CHECK-NEXT:    }
// CHECK-DAG:     return %[[V0]] : f32
func.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: i1) -> f32 {
  %0 = "tt.atomic_rmw" (%arg0, %arg1, %arg2) {atomic_rmw_op = 5 : i32, sem = 2 : i32} : (!tt.ptr<f32>, f32, i1) -> f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x2xindex>,
// CHECK-SAME:      %[[ARG1:.*]]: tensor<2x2xf32>,
// CHECK-SAME:      %[[ARG2:.*]]: tensor<2x2xi1>) -> tensor<2x2xf32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.empty() : tensor<2x2xf32>
// CHECK-NEXT:    %[[V1:.*]] = scf.for %[[ARG3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG4:.*]] = %[[V0]]) -> (tensor<2x2xf32>) {
// CHECK-NEXT:      %[[V2:.*]] = scf.for %[[ARG5:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG6:.*]] = %[[ARG4]]) -> (tensor<2x2xf32>) {
// CHECK-DAG:         %[[V3:.*]] = tensor.extract %[[ARG0]][%[[ARG3]], %[[ARG5]]] : tensor<2x2xindex>
// CHECK-DAG:         %[[V4:.*]] = tensor.extract %[[ARG1]][%[[ARG3]], %[[ARG5]]] : tensor<2x2xf32>
// CHECK-DAG:         %[[V5:.*]] = tensor.extract %[[ARG2]][%[[ARG3]], %[[ARG5]]] : tensor<2x2xi1>
// CHECK-DAG:         %[[V6:.*]] = arith.index_cast %[[V3]] : index to i64
// CHECK-DAG:         %[[V7:.*]] = llvm.inttoptr %[[V6]] : i64 to !llvm.ptr<f32, 1>
// CHECK-NEXT:        %[[V8:.*]] = scf.if %[[V5]] -> (f32) {
// CHECK-DAG:           %[[V9:.*]] = llvm.atomicrmw fadd %[[V7]], %[[V4]] acquire : !llvm.ptr<f32, 1>, f32
// CHECK-DAG:           scf.yield %[[V9]] : f32
// CHECK-NEXT:        } else {
// CHECK-DAG:           %[[Va:.*]] = llvm.mlir.undef : f32
// CHECK-DAG:           scf.yield %[[Va]] : f32
// CHECK-NEXT:        }
// CHECK-DAG:         %[[Vb:.*]] = tensor.insert %[[V8]] into %[[ARG6]][%[[ARG3]], %[[ARG5]]] : tensor<2x2xf32>
// CHECK-DAG:         scf.yield %[[Vb]] : tensor<2x2xf32>
// CHECK-NEXT:      }
// CHECK-DAG:       scf.yield %[[V2]] : tensor<2x2xf32>
// CHECK-NEXT:    }
// CHECK-DAG:     return %[[V1]] : tensor<2x2xf32>
func.func public @kernel(%arg0: tensor<2x2x!tt.ptr<f32>>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xi1>) -> tensor<2x2xf32> {
  %0 = "tt.atomic_rmw" (%arg0, %arg1, %arg2) {atomic_rmw_op = 5 : i32, sem = 2 : i32} : (tensor<2x2x!tt.ptr<f32>>, tensor<2x2xf32>, tensor<2x2xi1>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>
}
