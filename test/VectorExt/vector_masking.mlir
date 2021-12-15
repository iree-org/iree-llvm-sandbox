// RUN: mlir-proto-opt %s -test-vector-masking-utils=masking -split-input-file | FileCheck %s

func @func_pred0(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: vector<16xi1>) {
  vector_ext.predicate(%arg2: vector<16xi1>) {
    %c0 = arith.constant 0 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = vector.transfer_read %arg0[%c0], %cst {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
    %1 = arith.addf %0, %0 : vector<16xf32>
    vector.transfer_write %1, %arg1[%c0] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
  }
  return
}

// CHECK-LABEL:   func @func_pred0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?xf32>, %[[VAL_1:.*]]: memref<?xf32>,
// CHECK-SAME:                     %[[MASK:.*]]: vector<16xi1>) {
// CHECK-NEXT:      %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:      %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:      %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]][%[[VAL_3]]], %[[VAL_4]], %[[MASK]] {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
// CHECK-NEXT:      %[[VAL_6:.*]] = arith.addf %[[VAL_5]], %[[VAL_5]] : vector<16xf32>
// CHECK-NEXT:      vector.transfer_write %[[VAL_6]], %[[VAL_1]][%[[VAL_3]]], %[[MASK]] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
// CHECK-NEXT:      return
// CHECK-NEXT:    }

