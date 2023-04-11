// RUN: mlir-proto-opt %s -test-vector-masking-utils=predication -split-input-file | FileCheck %s


func.func @func_pred0(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
%pred0: vector<16xi1>, %idx0 : index, %incoming: vector<16xi1>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0], %f0 {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
  %add = arith.addf %0, %0 : vector<16xf32>
  vector.transfer_write %add, %arg1[%c0] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
  return
}

// CHECK-LABEL:   func.func @func_pred0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?xf32>, %[[VAL_1:.*]]: memref<?xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: vector<16xi1>,
// CHECK-SAME:                     %[[VAL_3:.*]]: index,
// CHECK-SAME:                     %[[VAL_4:.*]]: vector<16xi1>) {
// CHECK:           vector_ext.predicate(%[[VAL_2]], [%[[VAL_3]]], %[[VAL_4]]) : vector<16xi1> {
// CHECK:           ^bb0(%[[VAL_5:.*]]: vector<16xi1>):
// CHECK:             %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:             %[[VAL_8:.*]] = vector.transfer_read %[[VAL_0]][%[[VAL_6]]], %[[VAL_7]] {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
// CHECK:             %[[VAL_9:.*]] = arith.addf %[[VAL_8]], %[[VAL_8]] : vector<16xf32>
// CHECK:             vector.transfer_write %[[VAL_9]], %[[VAL_1]][%[[VAL_6]]] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
// CHECK:           }
// CHECK:           return
// CHECK:         }
