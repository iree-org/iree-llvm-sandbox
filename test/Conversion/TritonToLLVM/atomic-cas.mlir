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


// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>) -> tensor<2x2xi32> {
// CHECK-DAG:     %[[V0:.*]] = tensor.reshape %{{.*}}(%{{.*}}) : (tensor<4xindex>, tensor<2xindex>) -> tensor<2x2xindex>
// CHECK-DAG:     %[[V1:.*]] = tensor.reshape %{{.*}}(%{{.*}}) : (tensor<4xi32>, tensor<2xindex>) -> tensor<2x2xi32>
// CHECK-DAG:     %[[V2:.*]] = tensor.empty() : tensor<2x2xi32>
// CHECK-NEXT:    %[[V3:.*]] = scf.for %[[ARG1:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG2:.*]] = %[[V2]]) -> (tensor<2x2xi32>) {
// CHECK-NEXT:      %[[V4:.*]] = scf.for %[[ARG3:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG4:.*]] = %[[ARG2]]) -> (tensor<2x2xi32>) {
// CHECK-DAG:         %[[V5:.*]] = tensor.extract %[[V0]][%[[ARG1]], %[[ARG3]]] : tensor<2x2xindex>
// CHECK-DAG:         %[[V6:.*]] = tensor.extract %[[V1]][%[[ARG1]], %[[ARG3]]] : tensor<2x2xi32>
// CHECK-DAG:         %[[V7:.*]] = tensor.extract %[[V1]][%[[ARG1]], %[[ARG3]]] : tensor<2x2xi32>
// CHECK-DAG:         %[[V8:.*]] = arith.index_cast %[[V5]] : index to i64
// CHECK-DAG:         %[[V9:.*]] = llvm.inttoptr %[[V8]] : i64 to !llvm.ptr<i32, 1>
// CHECK-DAG:         %[[Va:.*]] = llvm.cmpxchg %[[V9]], %[[V6]], %[[V7]] acq_rel acquire : !llvm.ptr<i32, 1>, i32
// CHECK-DAG:         %[[Vb:.*]] = llvm.extractvalue %[[Va]][0] : !llvm.struct<(i32, i1)>
// CHECK-DAG:         %[[Vc:.*]] = tensor.insert %[[Vb]] into %[[ARG4]][%[[ARG1]], %[[ARG3]]] : tensor<2x2xi32>
// CHECK-NEXT:        scf.yield %[[Vc]] : tensor<2x2xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V4]] : tensor<2x2xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:     return %[[V3]] : tensor<2x2xi32>
func.func public @kernel(%arg0: !tt.ptr<i32>) -> tensor<2x2xi32> {
  %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
  %1 = tt.splat %arg0 : (!tt.ptr<i32>) -> tensor<4x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<i32>>, tensor<4xi32>
  %3 = tt.view %2 : (tensor<4x!tt.ptr<i32>>) -> tensor<2x2x!tt.ptr<i32>>
  %4 = tt.view %0 : (tensor<4xi32>) -> tensor<2x2xi32>
  %5 = "tt.atomic_cas"(%3, %4, %4) <{sem = 4 : i32}> : (tensor<2x2x!tt.ptr<i32>>, tensor<2x2xi32>, tensor<2x2xi32>) -> tensor<2x2xi32>
  return %5 : tensor<2x2xi32>
}
