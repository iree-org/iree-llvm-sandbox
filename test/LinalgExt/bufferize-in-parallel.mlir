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


  func @matmul_on_tensors(%arg0: tensor<3x7xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg1: tensor<7x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = false}, %arg2: tensor<3x5xf32> {linalg.buffer_layout = affine_map<(d0, d1) -> (d0, d1)>, linalg.inplaceable = true}) -> tensor<3x5xf32> attributes {passthrough = ["noinline", ["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]} {
    %c5 = arith.constant 5 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = linalg.fill(%cst, %arg2) : f32, tensor<3x5xf32> -> tensor<3x5xf32> 
    %1 = linalg_ext.in_parallel %c2  -> (tensor<3x5xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      %2 = affine.apply affine_map<(d0) -> (d0 * 2)>(%arg3)
      %3 = affine.min affine_map<(d0) -> (d0 * -2 + 3, 2)>(%arg3)
      %4 = tensor.extract_slice %0[%2, 0] [%3, 5] [1, 1] : tensor<3x5xf32> to tensor<?x5xf32>
      %5 = tensor.extract_slice %arg0[%2, 0] [%3, 7] [1, 1] : tensor<3x7xf32> to tensor<?x7xf32>
      %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%5, %arg1 : tensor<?x7xf32>, tensor<7x5xf32>) outs(%4 : tensor<?x5xf32>) {
      ^bb0(%arg4: f32, %arg5: f32, %arg6: f32):  // no predecessors
        %8 = arith.mulf %arg4, %arg5 : f32
        %9 = arith.addf %arg6, %8 : f32
        linalg.yield %9 : f32
      } -> tensor<?x5xf32>
      %7 = tensor.cast %6 : tensor<?x5xf32> to tensor<?x?xf32>
      linalg_ext.perform_concurrently {
        linalg_ext.parallel_insert_slice %7 into %0[%2, %c0] [%3, %c5] [%c1, %c1] : tensor<?x?xf32> into tensor<3x5xf32>
      }
    }
    return %1 : tensor<3x5xf32>
  }
  