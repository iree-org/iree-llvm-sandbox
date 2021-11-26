// RUN: mlir-proto-opt %s -linalg-tile-to-in-parallel| FileCheck %s

// CHECK-DAG: #[[$CEIL_MAP:.*]] = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
// CHECK-DAG: #[[$MUL_MAP:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$SUB_MAP:.*]] = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK-DAG: #[[$ID1_MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$ID2_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: func @static_tile
//  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
//  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<?xf32>
func @static_tile(%chunk_size: index, %in: tensor<?xf32>, %out: tensor<?xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0: index

  // CHECK: %[[M:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?xf32>
  // CHECK: %[[CEIL:.*]] = affine.apply #[[$CEIL_MAP]](%[[M]])[%[[CHUNK_SIZE]]]
  // CHECK: linalg_ext.in_parallel %[[CEIL]] -> (tensor<?xf32>) {
  %0 = linalg_ext.tile %chunk_size outs(%out: tensor<?xf32>) -> (tensor<?xf32>) {

  // CHECK: ^bb0(%[[TIDX:.*]]: index
  // CHECK:    %[[OFFSET:.*]] = affine.apply #[[$MUL_MAP]](%[[TIDX]])[%[[CHUNK_SIZE]]]
  // CHECK:    %[[REST:.*]] = affine.apply #[[$SUB_MAP]](%[[M]], %[[OFFSET]])
  // CHECK:    %[[SIZE:.*]] = affine.min #[[$ID2_MAP]](%[[REST]], %[[CHUNK_SIZE]])
  // CHECK:    %[[O:.*]] = tensor.extract_slice %[[OUT]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> to tensor<?xf32>

    // TODO: one offset and one size per tensor?
    // If not necessary in the dense strided-array world, what about the rest?
    ^bb0(%offset: index, %size: index, %st1: tensor<?xf32>):

      // TODO: atm this is just 1-1: out-chunk-size -> in-size.
  // CHECK:    %[[I:.*]] = tensor.extract_slice %[[IN]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> to tensor<?xf32>
      %1 = tensor.extract_slice %in[%offset][%size][1] : tensor<?xf32> to tensor<?xf32>

  // CHECK:    %[[R:.*]] = linalg.generic {{.*}} ins(%[[I]] : tensor<?xf32>) outs(%[[O]] : tensor<?xf32>)
      %3 = linalg.generic {
           indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
           iterator_types = ["parallel"]}
         ins(%1: tensor<?xf32>) outs(%st1: tensor<?xf32>) {
         ^bb0(%a: f32, %b:f32):  // no predecessors
           %f42 = arith.constant 42.0: f32
           %tmp = arith.mulf %a, %f42: f32
           linalg.yield %tmp: f32
      } -> tensor<?xf32>

  // CHECK: linalg_ext.perform_concurrently {
  // CHECK:    linalg_ext.parallel_insert_slice %[[R]] into %[[OUT]][%[[OFFSET]]] [%[[SIZE]]] [{{.*}}] : tensor<?xf32> into tensor<?xf32>
      linalg_ext.tile_yield %3: tensor<?xf32> 
  }
  return %0: tensor<?xf32>
}
