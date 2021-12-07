// RUN: mlir-proto-opt %s -test-vector-masking-utils=predication -split-input-file | FileCheck %s

func @tiled_loop_1d(%arg0: tensor<?xf32>,
                    %arg1: tensor<?xf32>) -> tensor<?xf32> {
  %c0 = arith.constant 0 : index
  %c8 = arith.constant 8 : index
  %0 = tensor.dim %arg0, %c0 : tensor<?xf32>
  %1 = linalg.init_tensor [%0] : tensor<?xf32>
  %3 = linalg.tiled_loop (%arg2) = (%c0) to (%0) step (%c8) ins (%arg3 = %arg0: tensor<?xf32>, %arg4 = %arg1: tensor<?xf32>) outs (%arg5 = %1: tensor<?xf32>) {
    %4 = affine.min affine_map<(d0)[s0] -> (8, -d0 + s0)>(%arg2)[%0]
    %5 = tensor.extract_slice %arg3[%arg2] [%4] [1] : tensor<?xf32> to tensor<?xf32>
    %6 = affine.min affine_map<(d0)[s0] -> (8, -d0 + s0)>(%arg2)[%0]
    %7 = tensor.extract_slice %arg4[%arg2] [%6] [1] : tensor<?xf32> to tensor<?xf32>
    %8 = affine.min affine_map<(d0)[s0] -> (8, -d0 + s0)>(%arg2)[%0]
    %9 = tensor.extract_slice %arg5[%arg2] [%8] [1] : tensor<?xf32> to tensor<?xf32>
    %10 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
                          iterator_types = ["parallel"]}
                          ins(%5, %7 : tensor<?xf32>, tensor<?xf32>) outs(%9 : tensor<?xf32>) {
    ^bb0(%arg6: f32, %arg7: f32, %arg8: f32):  // no predecessors
      %12 = arith.addf %arg6, %arg7 : f32
      linalg.yield %12 : f32
    } -> tensor<?xf32>
    %11 = tensor.insert_slice %10 into %arg5[%arg2] [%8] [1] : tensor<?xf32> into tensor<?xf32>
    linalg.yield %11 : tensor<?xf32>
  }
  return %3 : tensor<?xf32>
}

// CHECK-LABEL:   func @tiled_loop_1d(
// CHECK:           %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:           %[[STEP:.*]] = arith.constant 8 : index
// CHECK:           %[[UB:.*]] = tensor.dim
// CHECK:           linalg.tiled_loop (%[[IV:.*]]) = (%[[ZERO]]) to (%[[UB]]) step (%[[STEP]])
// CHECK-NEXT:        %[[PRED_MIN:.*]] = affine.min #map0(%[[UB]], %[[IV]])
// CHECK-NEXT:        %[[PRED:.*]] = vector.create_mask %[[PRED_MIN]] : vector<8xi1>
// CHECK-NEXT:        %[[PRED_RES:.*]] = vector_ext.predicate(%[[PRED]]: vector<8xi1>) -> (tensor<?xf32>) {
// CHECK-NEXT:          %[[VAL_15:.*]] = affine.min #map1(%[[IV]])[%[[UB]]]
// CHECK-NEXT:          %[[VAL_16:.*]] = tensor.extract_slice %{{.*}}[%[[IV]]] [%[[VAL_15]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK-NEXT:          %[[VAL_17:.*]] = affine.min #map1(%[[IV]])[%[[UB]]]
// CHECK-NEXT:          %[[VAL_18:.*]] = tensor.extract_slice %{{.*}}[%[[IV]]] [%[[VAL_17]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK-NEXT:          %[[VAL_19:.*]] = affine.min #map1(%[[IV]])[%[[UB]]]
// CHECK-NEXT:          %[[VAL_20:.*]] = tensor.extract_slice %{{.*}}[%[[IV]]] [%[[VAL_19]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK-NEXT:          %[[VAL_21:.*]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%[[VAL_16]], %[[VAL_18]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_20]] : tensor<?xf32>) {
// CHECK-NEXT:          ^bb0(%[[VAL_22:.*]]: f32, %[[VAL_23:.*]]: f32, %[[VAL_24:.*]]: f32):
// CHECK-NEXT:            %[[VAL_25:.*]] = arith.addf %[[VAL_22]], %[[VAL_23]] : f32
// CHECK-NEXT:            linalg.yield %[[VAL_25]] : f32
// CHECK-NEXT:          } -> tensor<?xf32>
// CHECK-NEXT:          %[[VAL_26:.*]] = tensor.insert_slice %[[VAL_21]] into %{{.*}}[%[[IV]]] [%[[VAL_19]]] [1] : tensor<?xf32> into tensor<?xf32>
// CHECK-NEXT:          vector_ext.yield %[[VAL_26]] : tensor<?xf32>
// CHECK-NEXT:        }
// CHECK-NEXT:        linalg.yield %[[PRED_RES:.*]] : tensor<?xf32>

// -----

func @func_pred0(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                 %pred0: vector<16xi1>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0], %f0 {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
  %add = arith.addf %0, %0 : vector<16xf32>
  vector.transfer_write %add, %arg1[%c0] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
  return
}

// CHECK-LABEL:   func @func_pred0(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<?xf32>, %[[VAL_1:.*]]: memref<?xf32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: vector<16xi1>) {
// CHECK-NEXT:      vector_ext.predicate(%[[VAL_2]]: vector<16xi1>) {
// CHECK-NEXT:        %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK-NEXT:        %[[VAL_4:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:        %[[VAL_5:.*]] = vector.transfer_read %[[VAL_0]]{{\[}}%[[VAL_3]]], %[[VAL_4]] {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
// CHECK-NEXT:        %[[VAL_6:.*]] = arith.addf %[[VAL_5]], %[[VAL_5]] : vector<16xf32>
// CHECK-NEXT:        vector.transfer_write %[[VAL_6]], %[[VAL_1]]{{\[}}%[[VAL_3]]] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
// CHECK-NEXT:      }
// CHECK-NEXT:      return
// CHECK-NEXT:    }
