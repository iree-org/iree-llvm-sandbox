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
