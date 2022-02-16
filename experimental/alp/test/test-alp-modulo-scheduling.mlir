// RUN: mlir-proto-opt -alp-modulo-scheduling %s | FileCheck %s 
#map1 = affine_map<()[s0] -> (s0 ceildiv 8)>

func @kernel(%arg0: memref<2048x256xf32>, %arg1: index, %arg2: memref<256x256x1x8xf32>, %arg3: index, %arg4: memref<32x256x1x8xf32>, %arg5: index){
  %0 = affine.apply #map1()[%arg5]
  %cst = arith.constant dense<0.000000e+00> : vector<4x4xf32>
  %c0 = arith.constant 0 : index
  %c256 = arith.constant 256 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %v0 = arith.constant dense<1.0> : vector<4x4xf32>
  %v1 = arith.constant dense<1.0> : vector<4x4xf32>
  %v2 = arith.constant dense<1.0> : vector<4x4xf32>
  %v3 = arith.constant dense<1.0> : vector<4x4xf32>

  // CHECK: vector.load %arg2
  // CHECK-NEXT: vector.load %arg2
  // CHECK-NEXT: vector.load %arg4
  // CHECK-NEXT: vector.load %arg4
  %41:4 = scf.for %arg6 = %c0 to %c256 step %c1 iter_args(%arg7 = %v0, %arg8 = %v1, %arg9 = %v2, %arg10 = %v3) -> (vector<4x4xf32>, vector<4x4xf32>, vector<4x4xf32>, vector<4x4xf32>) {
    // CHECK: scf.for
    // CHECK: %[[V0:.*]] = vector.load %arg2
    // CHECK-NEXT: %[[V1:.*]] = vector.load %arg2
    // CHECK-NEXT: %[[V2:.*]] = vector.load %arg4
    // CHECK-NEXT: %[[V3:.*]] = vector.load %arg4
    // CHECK-NEXT: vector.outerproduct %arg11, %arg12
    // CHECK-NEXT: vector.outerproduct %arg11, %arg13
    // CHECK-NEXT: vector.outerproduct %arg14, %arg12
    // CHECK-NEXT: vector.outerproduct %arg14, %arg13
    // CHECK: %[[Y0:.*]] = vector.load %arg2
    // CHECK: %[[Y1:.*]] = vector.load %arg2
    // CHECK: %[[Y2:.*]] = vector.load %arg4
    // CHECK: %[[Y3:.*]] = vector.load %arg4
    // CHECK: %[[Y4:.*]] = vector.outerproduct %[[V0]], %[[V2]]
    // CHECK-NEXT: %[[Y5:.*]] = vector.outerproduct %[[V0]], %[[V3]]
    // CHECK-NEXT: %[[Y6:.*]] = vector.outerproduct %[[V1]], %[[V2]]
    // CHECK-NEXT: %[[Y7:.*]] = vector.outerproduct %[[V1]], %[[V3]]
    // CHECK-NEXT: scf.yield %[[Y7]], %[[Y6]], %[[Y5]], %[[Y4]], %[[Y0]], %[[Y2]], %[[Y3]], %[[Y1]]
    %58 = vector.load %arg2[%arg3, %arg6, %c0, %c0] : memref<256x256x1x8xf32>, vector<4xf32>
    %59 = vector.load %arg2[%arg3, %arg6, %c0, %c4] : memref<256x256x1x8xf32>, vector<4xf32>
    %60 = vector.load %arg4[%0, %arg6, %c0, %c0] : memref<32x256x1x8xf32>, vector<4xf32>
    %61 = vector.load %arg4[%0, %arg6, %c0, %c4] : memref<32x256x1x8xf32>, vector<4xf32>
    %62 = vector.outerproduct %58, %60, %arg10 {kind = #vector.kind<add>} : vector<4xf32>, vector<4xf32>
    %63 = vector.outerproduct %58, %61, %arg9 {kind = #vector.kind<add>} : vector<4xf32>, vector<4xf32>
    %64 = vector.outerproduct %59, %60, %arg8 {kind = #vector.kind<add>} : vector<4xf32>, vector<4xf32>
    %65 = vector.outerproduct %59, %61, %arg7 {kind = #vector.kind<add>} : vector<4xf32>, vector<4xf32>
    scf.yield %65, %64, %63, %62 : vector<4x4xf32>, vector<4x4xf32>, vector<4x4xf32>, vector<4x4xf32>
  }
  %s0 = vector.extract %41#3[0] : vector<4x4xf32>
  vector.store %s0, %arg0[%c0, %c0] : memref<2048x256xf32>, vector<4xf32>

  return
}
