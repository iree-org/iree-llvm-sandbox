// RUN: mlir-proto-opt %s -linalg-bufferization-driver

#map0 = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0, d1) -> (d0 - d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>

func @static_tile(
    %arg0: index,
    %arg1: tensor<?xf32> {linalg.inplaceable=true},
    %arg2: tensor<?xf32> {linalg.inplaceable=true}) -> (tensor<?xf32>, f32)
{
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg2, %c0 : tensor<?xf32>
  %1 = affine.apply #map0(%0)[%arg0]

  %2 = linalg_ext.in_parallel %1  -> (tensor<?xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.apply #map2(%0, %3)
      %5 = affine.min #map3(%4, %arg0)

      %6 = tensor.extract_slice %arg2[%3] [%5] [%c1] : tensor<?xf32> to tensor<?xf32>
      // TODO: InParallel op should bufferize out-of-place instead of this one.
      %7 = tensor.extract_slice %arg1[%3] [%5] [1] : tensor<?xf32> to tensor<?xf32>

      // Note: Bufferizes inplace with the input.
      %8 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%7 : tensor<?xf32>) outs(%6 : tensor<?xf32>) {
      ^bb0(%arg4: f32, %arg5: f32):  // no predecessors
        %9 = arith.mulf %arg4, %cst : f32
        linalg.yield %9 : f32
      } -> tensor<?xf32>

      linalg_ext.perform_concurrently {
        linalg_ext.parallel_insert_slice %8 into %arg2[%3] [%5] [%c1] : tensor<?xf32> into tensor<?xf32>
      }
  }

  %f = tensor.extract %arg1[%c0] : tensor<?xf32>

  return %2, %f : tensor<?xf32>, f32
}
