// RUN: mlir-proto-opt %s | FileCheck %s

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
//       CHECK: func @reverse_1d_tensor
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]] : tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = linalg_ext.reverse
//  CHECK-SAME:       dimensions(dense<0> : tensor<1xi64>)
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<?x?xf32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:   return %[[RESULT]]

// -----

func @reverse_2d_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %reverse = linalg_ext.reverse
      dimensions(dense<[0, 1]> : tensor<2xi64>)
      ins(%arg0 : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>
  return %reverse : tensor<?x?xf32>
}
//       CHECK: func @reverse_2d_tensor
//  CHECK-SAME:      %[[ARG0:[a-zA-Z0-9]+]]: tensor<?x?xf32>
//   CHECK-DAG:   %[[C0:.+]] = arith.constant 0 : index
//   CHECK-DAG:   %[[C1:.+]] = arith.constant 1 : index
//   CHECK-DAG:   %[[D0:.+]] = tensor.dim %[[ARG0]], %[[C0]]
//   CHECK-DAG:   %[[D1:.+]] = tensor.dim %[[ARG0]], %[[C1]]
//       CHECK:   %[[INIT:.+]] = linalg.init_tensor [%[[D0]], %[[D1]]] : tensor<?x?xf32>
//       CHECK:   %[[RESULT:.+]] = linalg_ext.reverse
//  CHECK-SAME:       dimensions(dense<[0, 1]> : tensor<2xi64>)
//  CHECK-SAME:       ins(%[[ARG0]] : tensor<?x?xf32>)
//  CHECK-SAME:       outs(%[[INIT]] : tensor<?x?xf32>)
//       CHECK:   return %[[RESULT]]
