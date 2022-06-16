// RUN: mlir-proto-opt %s -test-vector-masking-utils=masking -split-input-file | FileCheck %s

func.func @func_pred0(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %pred_mask: vector<16xi1>,
                 %idx: index, %incoming_mask: vector<16xi1> ) {
  vector_ext.predicate(%pred_mask, [%idx], %incoming_mask): vector<16xi1> {
  ^bb0(%true_mask: vector<16xi1>):
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
    %1 = arith.addf %0, %0 : vector<16xf32>
    vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
  }
  return
}

// CHECK-LABEL:   func.func @func_pred0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?xf32>, %[[VAL_1:.*]]: memref<?xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: vector<16xi1>,
// CHECK-SAME:                     %[[VAL_3:.*]]: index,
// CHECK-SAME:                     %[[VAL_4:.*]]: vector<16xi1>) {
// CHECK:           %[[VAL_5:.*]] = arith.andi %[[VAL_4]], %[[VAL_2]] : vector<16xi1>
// CHECK:           %[[VAL_6:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_7:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = vector.transfer_read %[[VAL_0]][%[[VAL_6]]], %[[VAL_7]], %[[VAL_5]] {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
// CHECK:           %[[VAL_9:.*]] = arith.addf %[[VAL_8]], %[[VAL_8]] : vector<16xf32>
// CHECK:           vector.transfer_write %[[VAL_9]], %[[VAL_1]][%[[VAL_6]]], %[[VAL_5]] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
// CHECK:           return
// CHECK:         }

