// RUN: mlir-proto-opt -linalg-ext-tiling="tile-sizes=10,20" -split-input-file -canonicalize -cse %s | FileCheck %s

func @reverse_1d_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %reverse = linalg_ext.reverse
      dimensions(dense<0> : tensor<1xi64>)
      ins(%arg0 : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>
  return %reverse : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//      CHECK: func @reverse_1d_tensor
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C10:.+]] = arith.constant 10 : index
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[D0]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[ARG1:[a-zA-Z0-9]+]] = %[[INIT]])
//      CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[D1]] step %[[C20]]
// CHECK-SAME:         iter_args(%[[ARG2:[a-zA-Z0-9]+]] = %[[ARG1]])
//  CHECK-DAG:       %[[T0:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[D0]]]
//  CHECK-DAG:       %[[T1:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[D1]]]
//      CHECK:       %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]] [%[[T0]], %[[T1]]]
//      CHECK:       %[[OUT_SLICE:.+]] = linalg.init_tensor [%[[T0]], %[[T1]]]
//      CHECK:       %[[RESULT_SLICE:.+]] = linalg_ext.reverse
// CHECK-SAME:           __internal_linalg_transform__ = "tiled"
// CHECK-SAME:           ins(%[[INPUT_SLICE]]
// CHECK-SAME:           outs(%[[OUT_SLICE]]
//      CHECK:       %[[INSERT_SLICE:.+]] = tensor.insert_slice %[[RESULT_SLICE]] into %[[ARG2]][%[[IV0]], %[[IV1]]] [%[[T0]], %[[T1]]]
//      CHECK:       scf.yield %[[INSERT_SLICE]]
//      CHECK:     scf.yield %[[YIELD]]
//      CHECK:   return %[[RESULT]]
