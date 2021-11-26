// RUN: mlir-proto-opt %s | mlir-proto-opt | FileCheck %s

// CHECK-LABEL: func @static_tile
func @static_tile(%chunk_size: index, %in: tensor<?xf32>, %out: tensor<?xf32>, %out2: tensor<?xf32>) -> (tensor<?xf32>) {
  %c0 = arith.constant 0: index
  //%d0 = tensor.dim %out, %c0: tensor<?xf32>

  // CHECK: linalg_ext.tile %{{.*}} outs(%{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?xf32>)
  // CHECK: ^bb0(%{{.*}}: index, %{{.*}}: index, %{{.*}}: tensor<?xf32>, %{{.*}}: tensor<?xf32>):
  %0:2 = linalg_ext.tile %chunk_size outs(%out: tensor<?xf32>, %out2: tensor<?xf32>)
      -> (tensor<?xf32>, tensor<?xf32>) {
    // TODO: one offset and one size per tensor?
    // If not necessary in the dense strided-array world, what about the rest?
    ^bb0(%offset: index, %size: index, %st1: tensor<?xf32>, %st2: tensor<?xf32>):
      // TODO: atm this is just 1-1: out-chunk-size -> in-size.
      %1 = tensor.extract_slice %in[%offset][%size][1] : tensor<?xf32> to tensor<?xf32>
      %3 = linalg.generic {
           indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
           iterator_types = ["parallel"]}
         ins(%1: tensor<?xf32>) outs(%st1: tensor<?xf32>) {
         ^bb0(%a: f32, %b:f32):  // no predecessors
           %f42 = arith.constant 42.0: f32
           %tmp = arith.mulf %a, %f42: f32
           linalg.yield %tmp: f32
      } -> tensor<?xf32>
      linalg_ext.tile_yield %3, %st2: tensor<?xf32>, tensor<?xf32> // assumes dim is 0 and stacks
  }
  return %0#0: tensor<?xf32>
}
