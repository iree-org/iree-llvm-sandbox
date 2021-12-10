// RUN: mlir-proto-opt %s -linalg-in-parallel-to-sequential-for | FileCheck %s

// CHECK-DAG: #[[$MUL_MAP:.*]] = affine_map<(d0)[s0] -> (d0 * s0)>
// CHECK-DAG: #[[$SUB_MAP:.*]] = affine_map<(d0, d1) -> (d0 - d1)>
// CHECK-DAG: #[[$ID1_MAP:.*]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: #[[$ID2_MAP:.*]] = affine_map<(d0, d1) -> (d0, d1)>
#map0 = affine_map<(d0)[s0] -> (d0 ceildiv s0)>
#map1 = affine_map<(d0)[s0] -> (d0 * s0)>
#map2 = affine_map<(d0, d1) -> (d0 - d1)>
#map3 = affine_map<(d0, d1) -> (d0, d1)>
#map4 = affine_map<(d0) -> (d0)>

// CHECK-LABEL: func @static_tile_tensors
//  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
//  CHECK-SAME:   %[[IN:[0-9a-z]+]]: tensor<?xf32>
//  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: tensor<?xf32>
func @static_tile_tensors(%arg0: index, %arg1: tensor<?xf32>, %arg2: tensor<?xf32>) -> tensor<?xf32> {
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = tensor.dim %arg2, %c0 : tensor<?xf32>
  %1 = affine.apply #map0(%0)[%arg0]

  // CHECK: %[[M:.*]] = tensor.dim %{{.*}}, %{{.*}} : tensor<?xf32>
  // CHECK: scf.for %[[IV:.*]] = {{.*}} iter_args(%[[OUT:.*]] = %{{.*}}) -> (tensor<?xf32>) {
  %2 = linalg_ext.in_parallel %1  -> (tensor<?xf32>) {
    ^bb0(%arg3: index):  // no predecessors
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.apply #map2(%0, %3)
      %5 = affine.min #map3(%4, %arg0)

      // Check the iter_arg is properly propagated
      // CHECK:    %[[O:.*]] = tensor.extract_slice %[[OUT]][{{.*}}] : tensor<?xf32> to tensor<?xf32>
      %6 = tensor.extract_slice %arg2[%3] [%5] [%c1] : tensor<?xf32> to tensor<?xf32>
      %7 = tensor.extract_slice %arg1[%3] [%5] [1] : tensor<?xf32> to tensor<?xf32>

      %8 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%7 : tensor<?xf32>) outs(%6 : tensor<?xf32>) {
      ^bb0(%arg4: f32, %arg5: f32):  // no predecessors
        %9 = arith.mulf %arg4, %cst : f32
        linalg.yield %9 : f32
      } -> tensor<?xf32>

      // CHECK:    %[[RES:.*]] = tensor.insert_slice %{{.*}} into %[[OUT]][{{.*}}] : tensor<?xf32> into tensor<?xf32>
      // CHECK:    scf.yield %[[RES]] : tensor<?xf32>
      linalg_ext.perform_concurrently {
        linalg_ext.parallel_insert_slice %8 into %arg2[%3] [%5] [%c1] : tensor<?xf32> into tensor<?xf32>
      }
  }
  return %2 : tensor<?xf32>
}

// CHECK-LABEL: func @static_tile_buffers
//  CHECK-SAME:   %[[CHUNK_SIZE:[0-9a-z]+]]: index
//  CHECK-SAME:   %[[IN:[0-9a-z]+]]: memref<?xf32>
//  CHECK-SAME:   %[[OUT:[0-9a-z]+]]: memref<?xf32>
func @static_tile_buffers(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %cst = arith.constant 4.200000e+01 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %0 = memref.dim %arg2, %c0 : memref<?xf32>
  %1 = affine.apply #map0(%0)[%arg0]

  // CHECK: %[[C1:.*]] = arith.constant 1 : index
  // CHECK: %[[M:.*]] = memref.dim %{{.*}}, %{{.*}} : memref<?xf32>
  // CHECK: scf.for %[[IV:.*]] = {{.*}} step %[[C1]] {
  linalg_ext.in_parallel %1 -> () {
    ^bb0(%arg3: index):  // no predecessors
      %3 = affine.apply #map1(%arg3)[%arg0]
      %4 = affine.apply #map2(%0, %3)
      %5 = affine.min #map3(%4, %arg0)

      %6 = memref.subview %arg2[%3] [%5] [%c1] : memref<?xf32> to memref<?xf32, offset:?, strides:[?]>
      %7 = memref.subview %arg1[%3] [%5] [1] : memref<?xf32> to memref<?xf32, offset:?, strides:[?]>

      linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} 
        ins(%7 : memref<?xf32, offset:?, strides:[?]>) outs(%6 : memref<?xf32, offset:?, strides:[?]>) {
      ^bb0(%arg4: f32, %arg5: f32):  // no predecessors
        %9 = arith.mulf %arg4, %cst : f32
        linalg.yield %9 : f32
      }

      // Nothing is yielded, skip the terminator.
      // CHECK-NOT: scf.yield
      linalg_ext.perform_concurrently {
      }
  }
  return
}