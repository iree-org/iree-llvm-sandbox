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
// CHECK-SAME:                        %[[VAL_0:.*]]: tensor<?xf32>, %[[VAL_1:.*]]: tensor<?xf32>) -> tensor<?xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 8 : index
// CHECK:           %[[VAL_4:.*]] = tensor.dim %[[VAL_0]], %[[VAL_2]] : tensor<?xf32>
// CHECK:           %[[VAL_5:.*]] = linalg.init_tensor [%[[VAL_4]]] : tensor<?xf32>
// CHECK:           %[[VAL_6:.*]] = linalg.tiled_loop (%[[VAL_7:.*]]) = (%[[VAL_2]]) to (%[[VAL_4]]) step (%[[VAL_3]]) ins (%[[VAL_8:.*]] = %[[VAL_0]]: tensor<?xf32>, %[[VAL_9:.*]] = %[[VAL_1]]: tensor<?xf32>) outs (%[[VAL_10:.*]] = %[[VAL_5]]: tensor<?xf32>) {
// CHECK:             %[[VAL_11:.*]] = affine.min #map0(%[[VAL_4]], %[[VAL_7]])
// CHECK:             %[[VAL_12:.*]] = vector.create_mask %[[VAL_11]] : vector<8xi1>
// CHECK:             %[[VAL_13:.*]] = arith.constant dense<true> : vector<8xi1>
// CHECK:             %[[VAL_14:.*]] = vector_ext.predicate(%[[VAL_12]], [%[[VAL_7]]], %[[VAL_13]]) : vector<8xi1> -> (tensor<?xf32>) {
// CHECK:             ^bb0(%[[VAL_15:.*]]: vector<8xi1>):
// CHECK:               %[[VAL_16:.*]] = affine.min #map1(%[[VAL_7]])[%[[VAL_4]]]
// CHECK:               %[[VAL_17:.*]] = tensor.extract_slice %[[VAL_8]][%[[VAL_7]]] [%[[VAL_16]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:               %[[VAL_18:.*]] = affine.min #map1(%[[VAL_7]])[%[[VAL_4]]]
// CHECK:               %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_9]][%[[VAL_7]]] [%[[VAL_18]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:               %[[VAL_20:.*]] = affine.min #map1(%[[VAL_7]])[%[[VAL_4]]]
// CHECK:               %[[VAL_21:.*]] = tensor.extract_slice %[[VAL_10]][%[[VAL_7]]] [%[[VAL_20]]] [1] : tensor<?xf32> to tensor<?xf32>
// CHECK:               %[[VAL_22:.*]] = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%[[VAL_17]], %[[VAL_19]] : tensor<?xf32>, tensor<?xf32>) outs(%[[VAL_21]] : tensor<?xf32>) {
// CHECK:               ^bb0(%[[VAL_23:.*]]: f32, %[[VAL_24:.*]]: f32, %[[VAL_25:.*]]: f32):
// CHECK:                 %[[VAL_26:.*]] = arith.addf %[[VAL_23]], %[[VAL_24]] : f32
// CHECK:                 linalg.yield %[[VAL_26]] : f32
// CHECK:               } -> tensor<?xf32>
// CHECK:               %[[VAL_27:.*]] = tensor.insert_slice %[[VAL_28:.*]] into %[[VAL_10]][%[[VAL_7]]] [%[[VAL_20]]] [1] : tensor<?xf32> into tensor<?xf32>
// CHECK:               vector_ext.yield %[[VAL_27]] : tensor<?xf32>
// CHECK:             }
// CHECK:             linalg.yield %[[VAL_29:.*]] : tensor<?xf32>
// CHECK:           }
// CHECK:           return %[[VAL_30:.*]] : tensor<?xf32>
// CHECK:         }

// -----

func @func_pred0(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
%pred0: vector<16xi1>, %idx0 : index, %incoming: vector<16xi1>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0], %f0 {in_bounds = [true]} : memref<?xf32>, vector<16xf32>
  %add = arith.addf %0, %0 : vector<16xf32>
  vector.transfer_write %add, %arg1[%c0] {in_bounds = [true]} : vector<16xf32>, memref<?xf32>
  return
}

// CHECK-LABEL:   func @func_pred0(
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
