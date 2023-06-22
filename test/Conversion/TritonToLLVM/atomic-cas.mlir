// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32,
// CHECK-SAME:      %[[ARG2:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = llvm.cmpxchg %[[ARG0]], %[[ARG1]], %[[ARG2]] monotonic monotonic : !llvm.ptr<i32, 1>, i32
// CHECK-DAG:     %[[V1:.*]] = llvm.extractvalue %[[V0]][0] : !llvm.struct<(i32, i1)>
// CHECK-DAG:     return %[[V1]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32) -> i32 {
  %0 = "tt.atomic_cas"(%arg0, %arg1, %arg2) <{sem = 1 : i32}> : (!tt.ptr<i32>, i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32,
// CHECK-SAME:      %[[ARG2:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = llvm.cmpxchg %[[ARG0]], %[[ARG1]], %[[ARG2]] acquire acquire : !llvm.ptr<i32, 1>, i32
// CHECK-DAG:     %[[V1:.*]] = llvm.extractvalue %[[V0]][0] : !llvm.struct<(i32, i1)>
// CHECK-DAG:     return %[[V1]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32) -> i32 {
  %0 = "tt.atomic_cas"(%arg0, %arg1, %arg2) <{sem = 2 : i32}> : (!tt.ptr<i32>, i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32,
// CHECK-SAME:      %[[ARG2:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = llvm.cmpxchg %[[ARG0]], %[[ARG1]], %[[ARG2]] release acquire : !llvm.ptr<i32, 1>, i32
// CHECK-DAG:     %[[V1:.*]] = llvm.extractvalue %[[V0]][0] : !llvm.struct<(i32, i1)>
// CHECK-DAG:     return %[[V1]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32) -> i32 {
  %0 = "tt.atomic_cas"(%arg0, %arg1, %arg2) <{sem = 3 : i32}> : (!tt.ptr<i32>, i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32,
// CHECK-SAME:      %[[ARG2:.*]]: i32) -> i32 {
// CHECK-DAG:     %[[V0:.*]] = llvm.cmpxchg %[[ARG0]], %[[ARG1]], %[[ARG2]] acq_rel acquire : !llvm.ptr<i32, 1>, i32
// CHECK-DAG:     %[[V1:.*]] = llvm.extractvalue %[[V0]][0] : !llvm.struct<(i32, i1)>
// CHECK-DAG:     return %[[V1]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32, %arg2: i32) -> i32 {
  %0 = "tt.atomic_cas"(%arg0, %arg1, %arg2) <{sem = 4 : i32}> : (!tt.ptr<i32>, i32, i32) -> i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<i32, 1>, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG2:.*]]: !llvm.ptr<i32, 1>) -> !llvm.ptr<i32, 1> {
// CHECK-DAG:     %[[V0:.*]] = llvm.cmpxchg %[[ARG0]], %[[ARG1]], %[[ARG2]] acq_rel acquire : !llvm.ptr<ptr<i32, 1>, 1>, !llvm.ptr<i32, 1>
// CHECK-DAG:     %[[V1:.*]] = llvm.extractvalue %[[V0]][0] : !llvm.struct<(ptr<i32, 1>, i1)>
// CHECK-DAG:     return %[[V1]] : !llvm.ptr<i32, 1>
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<i32>>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<i32>) -> !tt.ptr<i32> {
  %0 = "tt.atomic_cas"(%arg0, %arg1, %arg2) <{sem = 4 : i32}> : (!tt.ptr<!tt.ptr<i32>>, !tt.ptr<i32>, !tt.ptr<i32>) -> !tt.ptr<i32>
  return %0 : !tt.ptr<i32>
}
