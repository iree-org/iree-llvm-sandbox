// RUN: mlir-proto-opt -alp-legalize %s | FileCheck %s 
#map0 = affine_map<(d0, d1)[s0, s1] -> (d0 * s1 + s0 + d1)>
#map1 = affine_map<()[s0] -> (s0 ceildiv 8)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d0)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
#map5 = affine_map<(d0) -> (d0 ceildiv 8)>
#map6 = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL: @kernel
func @kernel(%arg0: memref<2048x256xf32, #map0>, %arg1: index, %arg2: memref<256x256x1x8xf32>, %arg3: index, %arg4: memref<32x256x1x8xf32>, %arg5: index) attributes {passthrough = [["prefer-vector-width", "128"]]} {
  %cst = arith.constant 0.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %0 = affine.apply #map1()[%arg5]
  %1 = vector.transfer_read %arg0[%arg1, %arg5], %cst {in_bounds = [true, true]} : memref<2048x256xf32, #map0>, vector<8x8xf32>
  // CHECK: vector.transfer_read %arg0
  // CHECK: vector.transfer_read %arg0
  // CHECK: vector.transfer_read %arg0
  // CHECK: vector.transfer_read %arg0
  // CHECK: scf.for %[[it0:.*]] = %[[lb:.*]] to %[[ub:.*]] step %[[s:.*]] iter_args(
  // CHECK-SAME: %arg7 = %cst_0, %arg8 = %8, %arg9 = %5, %arg10 = %3, %arg11 = %1) -> (vector<8x8xf32>, vector<4x4xf32>, vector<4x4xf32>, vector<4x4xf32>, vector<4x4xf32>) {
  %2 = scf.for %arg6 = %c0 to %c256 step %c1 iter_args(%arg7 = %1) -> (vector<8x8xf32>) {
    // CHECK: %[[V0:.*]] = vector.transfer_read %arg2
    // CHECK-NEXT: %[[V1:.*]] = vector.transfer_read %arg2
    // CHECK-NEXT: %[[V2:.*]] = vector.transfer_read %arg4
    // CHECK-NEXT: %[[V3:.*]] = vector.transfer_read %arg4
    %3 = vector.transfer_read %arg2[%arg3, %arg6, %c0, %c0], %cst {in_bounds = [true, true]} : memref<256x256x1x8xf32>, vector<1x8xf32>
    %4 = vector.transfer_read %arg4[%0, %arg6, %c0, %c0], %cst {in_bounds = [true, true]} : memref<32x256x1x8xf32>, vector<1x8xf32>
    %5 = vector.contract {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %3, %4, %arg7 : vector<1x8xf32>, vector<1x8xf32> into vector<8x8xf32>
    scf.yield %5 : vector<8x8xf32>
  }
  vector.transfer_write %2, %arg0[%arg1, %arg5] {in_bounds = [true, true]} : vector<8x8xf32>, memref<2048x256xf32, #map0>
  return
}
