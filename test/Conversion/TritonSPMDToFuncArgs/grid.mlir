// RUN: structured-opt %s -split-input-file \
// RUN:   -convert-triton-spmd-to-func-args \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-NEXT:    return

// CHECK-LABEL: func.func public @kernel_grid(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32,
// CHECK-SAME:      %[[ARG3:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}
// CHECK-SAME:      %[[ARG4:.*]]: i32) {
// CHECK-DAG:     %[[V0:.*]] = arith.index_cast %[[ARG0]] : i32 to index
// CHECK-DAG:     %[[V1:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK-DAG:     %[[V2:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V4:.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.parallel (%[[ARG5:.*]], %[[ARG6:.*]], %[[ARG7:.*]]) = (%[[V3]], %[[V3]], %[[V3]]) to (%[[V0]], %[[V1]], %[[V2]]) step (%[[V4]], %[[V4]], %[[V4]]) {
// CHECK-DAG:       %[[V5:.*]] = arith.index_cast %[[ARG5]] : index to i32
// CHECK-DAG:       %[[V6:.*]] = arith.index_cast %[[ARG6]] : index to i32
// CHECK-DAG:       %[[V7:.*]] = arith.index_cast %[[ARG7]] : index to i32
// CHECK-DAG:       func.call @kernel(%[[V5]], %[[V6]], %[[V7]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]]) : (i32, i32, i32, i32, i32, i32, !tt.ptr<i32>, i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: i32) {
  return
}

// -----

// CHECK-LABEL: tt.func public @kernel(
// CHECK-NEXT:    tt.return

// CHECK-LABEL: tt.func public @kernel_grid(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32,
// CHECK-SAME:      %[[ARG3:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}
// CHECK-SAME:      %[[ARG4:.*]]: i32) {
// CHECK-DAG:     %[[V0:.*]] = arith.index_cast %[[ARG0]] : i32 to index
// CHECK-DAG:     %[[V1:.*]] = arith.index_cast %[[ARG1]] : i32 to index
// CHECK-DAG:     %[[V2:.*]] = arith.index_cast %[[ARG2]] : i32 to index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V4:.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.parallel (%[[ARG5:.*]], %[[ARG6:.*]], %[[ARG7:.*]]) = (%[[V3]], %[[V3]], %[[V3]]) to (%[[V0]], %[[V1]], %[[V2]]) step (%[[V4]], %[[V4]], %[[V4]]) {
// CHECK-DAG:       %[[V5:.*]] = arith.index_cast %[[ARG5]] : index to i32
// CHECK-DAG:       %[[V6:.*]] = arith.index_cast %[[ARG6]] : index to i32
// CHECK-DAG:       %[[V7:.*]] = arith.index_cast %[[ARG7]] : index to i32
// CHECK-DAG:       tt.call @kernel(%[[V5]], %[[V6]], %[[V7]], %[[ARG0]], %[[ARG1]], %[[ARG2]], %[[ARG3]], %[[ARG4]]) : (i32, i32, i32, i32, i32, i32, !tt.ptr<i32>, i32) -> ()
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    tt.return
tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: i32) {
  tt.return
}
