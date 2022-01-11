// RUN: mlir-proto-opt -linalg-ext-tiling="tile-sizes=10,20" %s | FileCheck %s

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
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<()[s0, s1, s2] -> (s0 - s1 - s2)>
//      CHECK: func @reverse_1d_tensor
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[C10:.+]] = arith.constant 10 : index
//  CHECK-DAG:   %[[C20:.+]] = arith.constant 20 : index
//  CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]]
//  CHECK-DAG:   %[[D0_0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[D1_0:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] = %[[C0]] to %[[D0_0]] step %[[C10]]
// CHECK-SAME:       iter_args(%[[ARG1:[a-zA-Z0-9]+]] = %[[INIT]])
//      CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] = %[[C0]] to %[[D1_0]] step %[[C20]]
// CHECK-SAME:         iter_args(%[[ARG2:[a-zA-Z0-9]+]] = %[[ARG1]])
//  CHECK-DAG:       %[[T0:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[D0_0]]]
//  CHECK-DAG:       %[[T1:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[D1_0]]]
//      CHECK:       %[[INPUT_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[IV1]]] [%[[T0]], %[[T1]]]
//      CHECK:       %[[D0_1:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//      CHECK:       %[[OFFSET:.+]] = affine.apply #[[MAP2]]()[%[[D0_1]], %[[IV0]], %[[T0]]]
//      CHECK:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[INIT]][%[[OFFSET]], %[[IV1]]] [%[[T0]], %[[T1]]]
//      CHECK:       %[[RESULT_SLICE:.+]] = linalg_ext.reverse
// CHECK-SAME:           ins(%[[INPUT_SLICE]]
// CHECK-SAME:           outs(%[[OUT_SLICE]]
//      CHECK:       %[[INSERT_SLICE:.+]] = tensor.insert_slice %[[RESULT_SLICE]] into %[[ARG2]][%[[IV0]], %[[IV1]]] [%[[T0]], %[[T1]]]
//      CHECK:       scf.yield %[[INSERT_SLICE]]
//      CHECK:     scf.yield %[[YIELD]]
//      CHECK:   return %[[RESULT]]

// -----

func @gemm(%lhs : tensor<?x?xf32>, %rhs : tensor<?x?xf32>,
    %init : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = linalg.matmul
      ins(%lhs, %rhs : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  return %0 : tensor<?x?xf32>
}
//  CHECK-DAG: #[[MAP0:.+]] = affine_map<(d0)[s0] -> (10, -d0 + s0)>
//  CHECK-DAG: #[[MAP1:.+]] = affine_map<(d0)[s0] -> (20, -d0 + s0)>
//  CHECK-DAG: #[[MAP2:.+]] = affine_map<(d0, d1)[s0] -> (d0, -d1 + s0)>
//  CHECK-DAG: #[[MAP3:.+]] = affine_map<()[s0] -> (s0, s0)>
//      CHECK: func @gemm
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9]+]]: tensor<?x?xf32>
// CHECK-SAME:     %[[ARG2:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//  CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//  CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//  CHECK-DAG:   %[[M:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:   %[[N:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//      CHECK:   %[[RESULT:.+]] = scf.for %[[IV0:[a-zA-Z0-9]+]] =
// CHECK-SAME:       iter_args(%[[ARG4:.+]] = %[[ARG2]])
//      CHECK:     %[[YIELD:.+]] = scf.for %[[IV1:[a-zA-Z0-9]+]] =
// CHECK-SAME:         iter_args(%[[ARG6:.+]] = %[[ARG4]])
//  CHECK-DAG:       %[[TILESIZE_Y:.+]] = affine.min #[[MAP0]](%[[IV0]])[%[[M]]]
//  CHECK-DAG:       %[[TILESIZE_X:.+]] = affine.min #[[MAP1]](%[[IV1]])[%[[N]]]
//  CHECK-DAG:       %[[M_1:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//  CHECK-DAG:       %[[K:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//  CHECK-DAG:       %[[N_1:.+]] = tensor.dim %[[ARG1]], %[[C1]]
//  CHECK-DAG:       %[[TILESIZE_Y_1:.+]] = affine.min #[[MAP2]](%[[TILESIZE_Y]], %[[IV0]])[%[[M_1]]]
//  CHECK-DAG:       %[[K_1:.+]] = affine.min #[[MAP3]]()[%[[K]]]
//  CHECK-DAG:       %[[LHS_SLICE:.+]] = tensor.extract_slice %[[ARG0]][%[[IV0]], %[[C0]]] [%[[TILESIZE_Y_1]], %[[K_1]]]
//  CHECK-DAG:       %[[K_2:.+]] = affine.min #[[MAP3]]()[%[[K]]]
//  CHECK-DAG:       %[[TILESIZE_X_1:.+]] = affine.min #[[MAP2]](%[[TILESIZE_X]], %[[IV1]])[%[[N_1]]]
//  CHECK-DAG:       %[[RHS_SLICE:.+]] = tensor.extract_slice %[[ARG1]][%[[C0]], %[[IV1]]] [%[[K_2]], %[[TILESIZE_X_1]]]
//  CHECK-DAG:       %[[TILESIZE_Y_2:.+]] = affine.min #[[MAP2]](%[[TILESIZE_Y]], %[[IV0]])[%[[M_1]]]
//  CHECK-DAG:       %[[TILESIZE_X_2:.+]] = affine.min #[[MAP2]](%[[TILESIZE_X]], %[[IV1]])[%[[N_1]]]
//  CHECK-DAG:       %[[OUT_SLICE:.+]] = tensor.extract_slice %[[ARG2]][%[[IV0]], %[[IV1]]] [%[[TILESIZE_Y_2]], %[[TILESIZE_X_2]]]
//      CHECK:       %[[RESULT_SLICE:.+]] = linalg.matmul
// CHECK-SAME:           ins(%[[LHS_SLICE]], %[[RHS_SLICE]]
// CHECK-SAME:           outs(%[[OUT_SLICE]]
//      CHECK:       %[[YIELD_INNER:.+]] = tensor.insert_slice %[[RESULT_SLICE]] into %[[ARG6]]
// CHECK-SAME:           [%[[IV0]], %[[IV1]]] [%[[TILESIZE_Y]], %[[TILESIZE_X]]]
//      CHECK:       scf.yield %[[YIELD_INNER]]
//      CHECK:     scf.yield %[[YIELD]]
//      CHECK:   return %[[RESULT]]
