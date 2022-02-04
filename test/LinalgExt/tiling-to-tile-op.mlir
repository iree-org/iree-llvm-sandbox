// RUN: mlir-proto-opt %s -linalg-ext-tiling-to-tile-op="tile-sizes=10" | FileCheck %s

// CHECK-LABEL: reverse_1d_tensor(
//  CHECK-SAME:   %[[T:[0-9a-z]+]]: tensor<?x?xf32>
func @reverse_1d_tensor(%arg0 : tensor<?x?xf32>) -> tensor<?x?xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>

  //      CHECK: %[[C10:.*]] = arith.constant 10 : index
  //      CHECK:   %[[I:.*]] = linalg.init_tensor {{.*}} : tensor<?x?xf32>
  //      CHECK: linalg_ext.tile %[[C10]] outs(%[[I]]: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[I_ITER:.*]]: tensor<?x?xf32>):
  //      CHECK:   %[[in:.*]] = tensor.extract_slice %[[T]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>) 
  // CHECK-SAME:     ins(%[[in]] : tensor<?x?xf32>) outs(%[[I_ITER]] : tensor<?x?xf32>) : tensor<?x?xf32>
  //      CHECK:   linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
  %reverse = linalg_ext.reverse
      dimensions(dense<0> : tensor<1xi64>)
      ins(%arg0 : tensor<?x?xf32>)
      outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>

  return %reverse : tensor<?x?xf32>
}

// CHECK-LABEL: matmul(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  //      CHECK: %[[C10:.*]] = arith.constant 10 : index
  //      CHECK: linalg_ext.tile %[[C10]] outs(%[[C]]: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[C_ITER:.*]]: tensor<?x?xf32>):
  //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tB:.*]] = tensor.extract_slice %[[B]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg.matmul 
  // CHECK-SAME:      ins(%[[tA]], %[[tB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[C_ITER]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  //      CHECK:   linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
  %D = linalg.matmul ins(%A, %B: tensor<?x?xf32>, tensor<?x?xf32>) outs(%C: tensor<?x?xf32>) -> tensor<?x?xf32>
  return %D: tensor<?x?xf32>
}

// CHECK-LABEL: matmul_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<100x200xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<200x300xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<100x300xf32>
func @matmul_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> (tensor<100x300xf32>) {
  //      CHECK: %[[C10:.*]] = arith.constant 10 : index
  //      CHECK: linalg_ext.tile %[[C10]] outs(%[[C]]: tensor<100x300xf32>) -> (tensor<100x300xf32>) {
  //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[C_ITER:.*]]: tensor<?x?xf32>):
  //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<100x200xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tB:.*]] = tensor.extract_slice %[[B]]{{.*}} : tensor<200x300xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg.matmul
  // CHECK-SAME:      ins(%[[tA]], %[[tB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[C_ITER]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  //      CHECK:   linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
  %D = linalg.matmul ins(%A, %B: tensor<100x200xf32>, tensor<200x300xf32>) outs(%C: tensor<100x300xf32>) -> tensor<100x300xf32>
  return %D: tensor<100x300xf32>
}
