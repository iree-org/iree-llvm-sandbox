// RUN: mlir-proto-opt %s -split-input-file -canonicalize -linalg-comprehensive-bufferize-inplace |\
// RUN: tee | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_matmul(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
func @add_matmul(
    %arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<32x64xf32>)
-> tensor<32x64xf32>
attributes { __writeable_func_buffer_args_attr__ = ["true", "true", "true"] }
{
  %cst = constant 0.000000e+00 : f32

  // CHECK-NOT:   memref.alloc
  // CHECK:   linalg.generic {{.*}} ins(%[[A]]{{.*}}) outs(%[[A]]{{.*}})
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0 : tensor<32x128xf32>, tensor<32x128xf32>)
   outs(%arg0 : tensor<32x128xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %3 = addf %arg3, %arg4 : f32
    linalg.yield %3 : f32
  } -> tensor<32x128xf32>

  // CHECK:   linalg.fill(%{{.*}}, %[[C]]) : f32, memref<32x64xf32>
  %1 = linalg.fill(%cst, %arg2) : f32, tensor<32x64xf32> -> tensor<32x64xf32>

  // CHECK:   linalg.matmul ins(%[[A]], %[[B]]{{.*}}) outs(%[[C]]{{.*}})
  %2 = linalg.matmul
    ins(%0, %arg1 : tensor<32x128xf32>, tensor<128x64xf32>)
   outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>
  return %2 : tensor<32x64xf32>
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: @add_matmul_needs_alloc(
//  CHECK-SAME:       %[[A:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[B:[0-9a-zA-Z]+]]: memref<
//  CHECK-SAME:       %[[C:[0-9a-zA-Z]+]]: memref<
func @add_matmul_needs_alloc(
    %arg0: tensor<32x128xf32>, %arg1: tensor<128x64xf32>, %arg2: tensor<32x64xf32>)
-> tensor<32x64xf32>
attributes { __writeable_func_buffer_args_attr__ = ["false", "true", "true"] }
{
  %cst = constant 0.000000e+00 : f32

  // CHECK:   %[[ALLOC:.*]] = memref.alloc
  // CHECK:   linalg.generic {{.*}} ins(%[[A]]{{.*}}) outs(%[[ALLOC]]{{.*}})
  %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
    ins(%arg0, %arg0 : tensor<32x128xf32>, tensor<32x128xf32>)
   outs(%arg0 : tensor<32x128xf32>) {
  ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):  // no predecessors
    %3 = addf %arg3, %arg4 : f32
    linalg.yield %3 : f32
  } -> tensor<32x128xf32>

  // CHECK:   linalg.fill(%{{.*}}, %[[C]]) : f32, memref<32x64xf32>
  %1 = linalg.fill(%cst, %arg2) : f32, tensor<32x64xf32> -> tensor<32x64xf32>

  // CHECK:   linalg.matmul ins(%[[ALLOC]], %[[B]]{{.*}}) outs(%[[C]]{{.*}})
  %2 = linalg.matmul
    ins(%0, %arg1 : tensor<32x128xf32>, tensor<128x64xf32>)
   outs(%1 : tensor<32x64xf32>) -> tensor<32x64xf32>

  // CHECK: memref.dealloc %[[ALLOC]]
  return %2 : tensor<32x64xf32>
}
