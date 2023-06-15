// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: i32)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>, %arg1: i32) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : i32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: f32)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<f32, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<f32>, %arg1: f32) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : f32
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<i32, 1>, 1>,
// CHECK-SAME:      %[[ARG1:.*]]: !llvm.ptr<i32, 1>)
// CHECK-NEXT:    llvm.store %[[ARG1]], %[[ARG0]] : !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<i32>>, %arg1: !tt.ptr<i32>) {
  tt.store %arg0, %arg1 {cache = 1 : i32, evict = 1 : i32} : !tt.ptr<i32>
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>)
// CHECK-DAG:     %[[V0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK-DAG:     %[[V1:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<2xindex>
// CHECK-DAG:     %[[V2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[V4:.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.for %[[ARG1:.*]] = %[[V2]] to %[[V3]] step %[[V4]] {
// CHECK-DAG:       %[[V5:.*]] = tensor.extract %[[V0]][%[[ARG1]]] : tensor<2xi32>
// CHECK-DAG:       %[[V6:.*]] = tensor.extract %[[V1]][%[[ARG1]]] : tensor<2xindex>
// CHECK-DAG:       %[[V7:.*]] = arith.index_cast %[[V6]] : index to i64
// CHECK-DAG:       %[[V8:.*]] = llvm.inttoptr %[[V7]] : i64 to !llvm.ptr<i32, 1>
// CHECK-NEXT:      llvm.store %[[V5]], %[[V8]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>) {
  %0 = arith.constant dense<0> : tensor<2xi32>
  %1 = tt.splat %arg0 : (!tt.ptr<i32>) -> tensor<2x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<i32>>, tensor<2xi32>
  tt.store %2, %0 {cache = 1 : i32, evict = 1 : i32} : tensor<2xi32>
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>)
// CHECK-DAG:     %[[V0:.*]] = arith.constant dense<0> : tensor<2xi32>
// CHECK-DAG:     %[[V1:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<2xindex>
// CHECK-DAG:     %[[V2:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[V4:.*]] = arith.constant 1 : index
// CHECK-NEXT:    scf.for %[[ARG1:.*]] = %[[V2]] to %[[V3]] step %[[V4]] {
// CHECK-DAG:       %[[V5:.*]] = tensor.extract %{{.*}}[%[[ARG1]]] : tensor<2xindex>
// CHECK-DAG:       %[[V6:.*]] = arith.index_cast %[[V5]] : index to i64
// CHECK-DAG:       %[[V7:.*]] = llvm.inttoptr %[[V6]] : i64 to !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-DAG:       %[[V8:.*]] = tensor.extract %[[V1]][%[[ARG1]]] : tensor<2xindex>
// CHECK-DAG:       %[[V9:.*]] = arith.index_cast %[[V8]] : index to i64
// CHECK-DAG:       %[[Va:.*]] = llvm.inttoptr %[[V9]] : i64 to !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-NEXT:      llvm.store %[[V7]], %[[Va]] : !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<i32>>, %arg1: !tt.ptr<i32>) {
  %0 = arith.constant dense<0> : tensor<2xi32>
  %1 = tt.splat %arg0 : (!tt.ptr<!tt.ptr<i32>>) -> tensor<2x!tt.ptr<!tt.ptr<i32>>>
  %2 = tt.addptr %1, %0 : tensor<2x!tt.ptr<!tt.ptr<i32>>>, tensor<2xi32>
  %3 = tt.splat %arg1 : (!tt.ptr<i32>) -> tensor<2x!tt.ptr<i32>>
  tt.store %2, %3 {cache = 1 : i32, evict = 1 : i32} : tensor<2x!tt.ptr<i32>>
  return
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>)
// CHECK-DAG:     %[[V0:.*]] = arith.constant dense<0> : tensor<2x8xi32>
// CHECK-DAG:     %[[V1:.*]] = arith.addi %{{.*}}, %{{.*}} : tensor<2x8xindex>
// CHECK-DAG:     %[[V2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 8 : index
// CHECK-DAG:     scf.for %[[ARG1:.*]] = %{{.*}} to %[[V2]] step %{{.*}} {
// CHECK-DAG:       scf.for %[[ARG2:.*]] = %{{.*}} to %[[V3]] step %{{.*}} {
// CHECK-DAG:         %[[V4:.*]] = tensor.extract %[[V0]][%[[ARG1]], %[[ARG2]]] : tensor<2x8xi32>
// CHECK-DAG:         %[[V5:.*]] = tensor.extract %[[V1]][%[[ARG1]], %[[ARG2]]] : tensor<2x8xindex>
// CHECK-DAG:         %[[V6:.*]] = arith.index_cast %[[V5]] : index to i64
// CHECK-DAG:         %[[V7:.*]] = llvm.inttoptr %[[V6]] : i64 to !llvm.ptr<i32, 1>
// CHECK-DAG:         llvm.store %[[V4]], %[[V7]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:      }
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func public @kernel(%arg0: !tt.ptr<i32>) {
  %0 = arith.constant dense<0> : tensor<2x8xi32>
  %1 = tt.splat %arg0 : (!tt.ptr<i32>) -> tensor<2x8x!tt.ptr<i32>>
  %2 = tt.addptr %1, %0 : tensor<2x8x!tt.ptr<i32>>, tensor<2x8xi32>
  tt.store %2, %0 {cache = 1 : i32, evict = 1 : i32} : tensor<2x8xi32>
  return
}
