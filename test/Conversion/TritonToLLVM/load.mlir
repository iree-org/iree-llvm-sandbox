// RUN: structured-opt %s \
// RUN:   -convert-triton-to-llvm -split-input-file \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<i32, 1>) -> i32 {
// CHECK-NEXT:    %[[V0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<i32, 1>
// CHECK-NEXT:    return %[[V0]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32>) -> i32 {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : i32
  return %0 : i32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<f32, 1>) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<f32, 1>
// CHECK-NEXT:    return %[[V0]] : f32
func.func public @kernel(%arg0: !tt.ptr<f32>) -> f32 {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
  return %0 : f32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: !llvm.ptr<ptr<f32, 1>, 1>) -> f32 {
// CHECK-NEXT:    %[[V0:.*]] = llvm.load %[[ARG0]] : !llvm.ptr<ptr<f32, 1>, 1>
// CHECK-NEXT:    %[[V1:.*]] = llvm.load %[[V0]] : !llvm.ptr<f32, 1>
// CHECK-NEXT:    return %[[V1]] : f32
func.func public @kernel(%arg0: !tt.ptr<!tt.ptr<f32>>) -> f32 {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : !tt.ptr<f32>
  %1 = tt.load %0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : f32
  return %1 : f32
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]:  tensor<2xindex>) -> tensor<2xi32> {
// CHECK-DAG:     %[[V1:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[V4:.*]] = tensor.empty() : tensor<2xi32>
// CHECK-NEXT:    %[[V5:.*]] = scf.for %[[ARG1:.*]] = %[[V1]] to %[[V2]] step %[[V3]] iter_args(%[[ARG2:.*]] = %[[V4]]) -> (tensor<2xi32>) {
// CHECK-DAG:       %[[V6:.*]] = tensor.extract %[[ARG0]][%[[ARG1]]] : tensor<2xindex>
// CHECK-DAG:       %[[V7:.*]] = arith.index_cast %[[V6]] : index to i64
// CHECK-DAG:       %[[V8:.*]] = llvm.inttoptr %[[V7]] : i64 to !llvm.ptr<i32, 1>
// CHECK-DAG:       %[[V9:.*]] = llvm.load %[[V8]] : !llvm.ptr<i32, 1>
// CHECK-DAG:       %[[Va:.*]] = tensor.insert %[[V9]] into %[[ARG2]][%[[ARG1]]] : tensor<2xi32>
// CHECK-NEXT:      scf.yield %[[Va]] : tensor<2xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func public @kernel(%arg0: tensor<2x!tt.ptr<i32>>) -> tensor<2xi32> {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2xi32>
  return %0 : tensor<2xi32>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2xindex>) -> tensor<2xindex> {
// CHECK-DAG:     %[[V1:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[V2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[V3:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[V4:.*]] = tensor.empty() : tensor<2xindex>
// CHECK-NEXT:    %[[V5:.*]] = scf.for %[[ARG1:.*]] = %[[V1]] to %[[V2]] step %[[V3]] iter_args(%[[ARG2:.*]] = %[[V4]]) -> (tensor<2xindex>) {
// CHECK-DAG:       %[[V6:.*]] = tensor.extract %[[ARG0]][%[[ARG1]]] : tensor<2xindex>
// CHECK-DAG:       %[[V7:.*]] = arith.index_cast %[[V6]] : index to i64
// CHECK-DAG:       %[[V8:.*]] = llvm.inttoptr %[[V7]] : i64 to !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-DAG:       %[[V9:.*]] = llvm.load %[[V8]] : !llvm.ptr<ptr<i32, 1>, 1>
// CHECK-DAG:       %[[Va:.*]] = llvm.ptrtoint %[[V9]] : !llvm.ptr<i32, 1> to i64
// CHECK-DAG:       %[[Vb:.*]] = arith.index_cast %[[Va]] : i64 to index
// CHECK-DAG:       %[[Vc:.*]] = tensor.insert %[[Vb]] into %[[ARG2]][%[[ARG1]]] : tensor<2xindex>
// CHECK-NEXT:      scf.yield %[[Vc]] : tensor<2xindex>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V5]] : tensor<2xindex>
func.func public @kernel(%arg0: tensor<2x!tt.ptr<!tt.ptr<i32>>>) -> tensor<2x!tt.ptr<i32>> {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2x!tt.ptr<i32>>
  return %0 : tensor<2x!tt.ptr<i32>>
}

// -----

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: tensor<2x8xindex>) -> tensor<2x8xi32> {
// CHECK-DAG:     %[[V1:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[V2:.*]] = arith.constant 8 : index
// CHECK-DAG:     %[[V3:.*]] = tensor.empty() : tensor<2x8xi32>
// CHECK-NEXT:    %[[V6:.*]] = scf.for %[[ARG1:.*]] = %{{.*}} to %[[V1]] step %{{.*}} iter_args(%[[ARG2:.*]] = %[[V3]]) -> (tensor<2x8xi32>) {
// CHECK-NEXT:      %[[V7:.*]] = scf.for %[[ARG3:.*]] = %{{.*}} to %[[V2]] step %{{.*}} iter_args(%[[ARG4:.*]] = %[[ARG2]]) -> (tensor<2x8xi32>) {
// CHECK-DAG:         %[[V8:.*]] = tensor.extract %[[ARG0]][%[[ARG1]], %[[ARG3]]] : tensor<2x8xindex>
// CHECK-DAG:         %[[V9:.*]] = arith.index_cast %[[V8]] : index to i64
// CHECK-DAG:         %[[Va:.*]] = llvm.inttoptr %[[V9]] : i64 to !llvm.ptr<i32, 1>
// CHECK-DAG:         %[[Vb:.*]] = llvm.load %[[Va]] : !llvm.ptr<i32, 1>
// CHECK-DAG:         %[[Vc:.*]] = tensor.insert %[[Vb]] into %[[ARG4]][%[[ARG1]], %arg3] : tensor<2x8xi32>
// CHECK-NEXT:        scf.yield %[[Vc]] : tensor<2x8xi32>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V7]] : tensor<2x8xi32>
// CHECK-NEXT:    }
// CHECK-NEXT:    return
func.func public @kernel(%arg0: tensor<2x8x!tt.ptr<i32>>) -> tensor<2x8xi32> {
  %0 = tt.load %arg0 {cache = 1 : i32, evict = 1 : i32, isVolatile = false} : tensor<2x8xi32>
  return %0 : tensor<2x8xi32>
}
