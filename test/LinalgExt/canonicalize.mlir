// RUN: mlir-proto-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func @canonicalize_insert_slice_indices(
//  CHECK-SAME:     %[[arg0:.*]]: tensor<?x?xf32>, %[[arg1:.*]]: tensor<?x?xf32>,
//  CHECK-SAME:     %[[idx:.*]]: index
func @canonicalize_insert_slice_indices(
    %arg0 : tensor<?x?xf32>, %arg1: tensor<?x?xf32>,
    %idx : index) -> tensor<?x?xf32>
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  %2 = linalg_ext.in_parallel %idx  -> (tensor<?x?xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      linalg_ext.perform_concurrently {
        // CHECK: linalg_ext.parallel_insert_slice %[[arg0]] into %arg1[%[[idx]], 0] [1, 5] [1, 1]
        linalg_ext.parallel_insert_slice %arg0 into %arg1[%idx, %c0] [%c1, 5] [%c1, %c1] : tensor<?x?xf32> into tensor<?x?xf32>
      }
  }
  return %2 : tensor<?x?xf32>
}
