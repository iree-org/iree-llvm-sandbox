// RUN: mlir-proto-opt -pass-pipeline="builtin.func(linalg-fuse{anchor-func=matmul anchor-op=linalg.generic tile-sizes=10,20,30 vectorize})" -split-input-file %s | FileCheck %s

func @matmul_bias_add(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2 : tensor<?xf32>) -> tensor<?x?xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
  %d1 = tensor.dim %arg1, %c1 : tensor<?x?xf32>
  %init = linalg.init_tensor [%d0, %d1] : tensor<?x?xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<?x?xf32>, tensor<?x?xf32>)
      outs(%0 : tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<?x?xf32>, tensor<?xf32>)
    outs(%init : tensor<?x?xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<?x?xf32>
  return %2 : tensor<?x?xf32>
}
//      CHECK: func @matmul_bias_add(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//      CHECK:       linalg.fill
//      CHECK:       linalg.matmul
//      CHECK:       linalg.generic
//      CHECK:     }
//      CHECK:   }

// -----

func @matmul_bias_add_static(%arg0 : tensor<20x60xf32>, %arg1 : tensor<60x120xf32>, %arg2 : tensor<120xf32>) -> tensor<20x120xf32> {
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %init = linalg.init_tensor [20, 120] : tensor<20x120xf32>
  %0 = linalg.fill ins(%cst : f32) outs(%init : tensor<20x120xf32>) -> tensor<20x120xf32>
  %1 = linalg.matmul ins(%arg0, %arg1 : tensor<20x60xf32>, tensor<60x120xf32>)
      outs(%0 : tensor<20x120xf32>) -> tensor<20x120xf32>
  %2 = linalg.generic {
    indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d1)>, affine_map<(d0, d1)-> (d0, d1)>],
    iterator_types = ["parallel", "parallel"]}
    ins(%1, %arg2 : tensor<20x120xf32>, tensor<120xf32>)
    outs(%init : tensor<20x120xf32>) {
      ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
        %3 = arith.addf %arg3, %arg4 : f32
        linalg.yield %3 : f32
    } -> tensor<20x120xf32>
  return %2 : tensor<20x120xf32>
}
// CHECK: func @matmul_bias_add_static(
//      CHECK:   scf.for
// CHECK-SAME:   {
//      CHECK:     scf.for
// CHECK-SAME:     {
//  CHECK-NOT:      linalg.fill
//  CHECK-NOT:      linalg.matmul
//  CHECK-NOT:      linalg.generic
//      CHECK:       %[[CONTRACT:.+]] = vector.contract
//      CHECK:       %[[UPDATE:.+]] = arith.addf %[[CONTRACT]], %{{.+}}
//      CHECK:       %[[BIAS:.+]] = vector.broadcast
//      CHECK:       %[[BIAS_ADD:.+]] = arith.addf %[[UPDATE]], %[[BIAS]]
//  CHECK-NOT:      linalg.fill
//  CHECK-NOT:      linalg.matmul
//  CHECK-NOT:      linalg.generic
//      CHECK:     }
//      CHECK:   }
