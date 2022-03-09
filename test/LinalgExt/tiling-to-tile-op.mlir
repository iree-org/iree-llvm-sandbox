// RUN: mlir-proto-opt %s -linalg-interp-transforms --split-input-file | FileCheck %s

module {
  // CHECK-LABEL: reverse_1d_tensor(
  //  CHECK-SAME:   %[[T:[0-9a-z]+]]: tensor<?x?xf32>
  func @reverse_1d_tensor(%A : tensor<?x?xf32>) -> tensor<?x?xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %d0 = tensor.dim %A, %c0 : tensor<?x?xf32>
    %d1 = tensor.dim %A, %c1 : tensor<?x?xf32>
    %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>

    //      CHECK: %[[C10:.*]] = arith.constant 10 : index
    //      CHECK:   %[[I:.*]] = linalg.init_tensor {{.*}} : tensor<?x?xf32>
    //      CHECK: linalg_ext.tile %[[C10]] outs(%[[I]]: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
    //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[I_ITER:.*]]: tensor<?x?xf32>):
    //      CHECK:   %[[in:.*]] = tensor.extract_slice %[[T]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
    //      CHECK:   %[[RES:.*]] = linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>) 
    // CHECK-SAME:     ins(%[[in]] : tensor<?x?xf32>) outs(%[[I_ITER]] : tensor<?x?xf32>) : tensor<?x?xf32>
    //      CHECK:   linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
    %reverse = linalg_ext.reverse dimensions(dense<0> : tensor<1xi64>)
         ins(%A : tensor<?x?xf32>)
        outs(%init : tensor<?x?xf32>) : tensor<?x?xf32>

    return %reverse : tensor<?x?xf32>
  }
  pdl.pattern @match_linalg_ext_reverse : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "linalg_ext.reverse"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
    rewrite %2 with "linalg_transform.apply"
  }
  linalg_transform.sequence {
    %0 = match @match_linalg_ext_reverse
    %1 = tile_to_linalg_ext_tile_op %0 {sizes = [10]}
  }
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1)[s0] -> (d0, -d1 + s0)>
module {
// CHECK-LABEL: matmul(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<?x?xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<?x?xf32>
  func @matmul(%A: tensor<?x?xf32>, %B: tensor<?x?xf32>, %C: tensor<?x?xf32>) -> tensor<?x?xf32> {
  //      CHECK: %[[C10:.*]] = arith.constant 10 : index
  //      CHECK: linalg_ext.tile %[[C10]] outs(%[[C]]: tensor<?x?xf32>) -> (tensor<?x?xf32>) {
  //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[C_ITER:.*]]: tensor<?x?xf32>):
  //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[tB:.*]] = tensor.extract_slice %[[B]]{{.*}} : tensor<?x?xf32> to tensor<?x?xf32>
  //      CHECK:   %[[RES:.*]] = linalg.matmul 
  // CHECK-SAME:      ins(%[[tA]], %[[tB]] : tensor<?x?xf32>, tensor<?x?xf32>)
  // CHECK-SAME:     outs(%[[C_ITER]] : tensor<?x?xf32>) -> tensor<?x?xf32>
  //      CHECK:   linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<?x?xf32>, tensor<?x?xf32>) 
                      outs(%C : tensor<?x?xf32>) -> (tensor<?x?xf32>)
    return %0 : tensor<?x?xf32>
  }
  pdl.pattern @match_linalg_matmul : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "linalg.matmul"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
    rewrite %2 with "linalg_transform.apply"
  }
  linalg_transform.sequence {
    %0 = match @match_linalg_matmul
    %1 = tile_to_linalg_ext_tile_op %0 {sizes = [10]}
  }
}

// -----

// CHECK: #[[$MAP:.+]] = affine_map<(d0, d1) -> (d0, -d1 + 100)>
module {
// CHECK-LABEL: matmul_static(
//  CHECK-SAME:   %[[A:[0-9a-z]+]]: tensor<100x200xf32>
//  CHECK-SAME:   %[[B:[0-9a-z]+]]: tensor<200x300xf32>
//  CHECK-SAME:   %[[C:[0-9a-z]+]]: tensor<100x300xf32>
  func @matmul_static(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>, %C: tensor<100x300xf32>) -> tensor<100x300xf32> {
    //      CHECK: %[[C10:.*]] = arith.constant 10 : index
    //      CHECK: linalg_ext.tile %[[C10]] outs(%[[C]]: tensor<100x300xf32>) -> (tensor<100x300xf32>) {
    //      CHECK: ^bb0(%[[OFF:.*]]: index, %[[SZ:.*]]: index, %[[C_ITER:.*]]: tensor<?x?xf32>):
    //      CHECK:   %[[M:.*]] = affine.min #[[$MAP]](%[[SZ]], %[[OFF]])
    //      CHECK:   %[[tA:.*]] = tensor.extract_slice %[[A]]{{.*}} : tensor<100x200xf32> to tensor<?x200xf32>
    //      CHECK:   %[[RES:.*]] = linalg.matmul
    // CHECK-SAME:      ins(%[[tA]], %[[B]] : tensor<?x200xf32>, tensor<200x300xf32>)
    // CHECK-SAME:     outs(%[[C_ITER]] : tensor<?x?xf32>) -> tensor<?x?xf32>
    //      CHECK:   linalg_ext.tile_yield %[[RES]] : tensor<?x?xf32>
    %0 = linalg.matmul ins(%A, %B : tensor<100x200xf32>, tensor<200x300xf32>) outs(%C : tensor<100x300xf32>) -> (tensor<100x300xf32>)
    return %0 : tensor<100x300xf32>
  }
  pdl.pattern @match_linalg_matmul : benefit(1) {
    %0 = operands
    %1 = types
    %2 = operation "linalg.matmul"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
    rewrite %2 with "linalg_transform.apply"
  }
  linalg_transform.sequence {
    %0 = match @match_linalg_matmul
    %1 = tile_to_linalg_ext_tile_op %0 {sizes = [10]}
  }
}
