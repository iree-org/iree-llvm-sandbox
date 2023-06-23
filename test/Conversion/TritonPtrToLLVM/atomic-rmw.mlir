// RUN: structured-opt %s \
// RUN:   -convert-triton-ptr-to-llvm -split-input-file \
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
