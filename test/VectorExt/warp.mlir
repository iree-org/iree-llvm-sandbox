// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute=propagate-distribution -canonicalize | FileCheck %s
// RUN: mlir-proto-opt %s -allow-unregistered-dialect -split-input-file -test-vector-warp-distribute=rewrite-warp-ops-to-scf-if -canonicalize | FileCheck %s --check-prefix=CHECK-SCF-IF

// CHECK-LABEL:   func @warp_dead_result(
func @warp_dead_result(%laneid: index) -> (vector<1xf32>) {
  // CHECK: %[[R:.*]] = vector_ext.warp_execute_on_lane_0(%{{.*}}) -> (vector<1xf32>)
    %r:3 = vector_ext.warp_execute_on_lane_0(%laneid) -> 
    (vector<1xf32>, vector<1xf32>, vector<1xf32>) {
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = "some_def"() : () -> (vector<32xf32>)
  // CHECK:   vector_ext.yield %{{.*}} : vector<32xf32>
    vector_ext.yield %2, %3, %4 : vector<32xf32>, vector<32xf32>, vector<32xf32>
  }
  // CHECK: return %[[R]] : vector<1xf32>
  return %r#1 : vector<1xf32>
}

// -----

#map0 = affine_map<()[s0] -> (s0 * 2)>

// CHECK-LABEL:   func @warp_propagate_elementwise(
func @warp_propagate_elementwise(%laneid: index, %dest: memref<1024xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  // CHECK: %[[R:.*]]:4 = vector_ext.warp_execute_on_lane_0(%{{.*}}) -> (vector<1xf32>, vector<1xf32>, vector<2xf32>, vector<2xf32>)
  %r:2 = vector_ext.warp_execute_on_lane_0(%laneid) -> 
    (vector<1xf32>, vector<2xf32>) {
    // CHECK: %[[V0:.*]] = "some_def"() : () -> vector<32xf32>
    // CHECK: %[[V1:.*]] = "some_def"() : () -> vector<32xf32>
    // CHECK: %[[V2:.*]] = "some_def"() : () -> vector<64xf32>
    // CHECK: %[[V3:.*]] = "some_def"() : () -> vector<64xf32>
    // CHECK: vector_ext.yield %[[V0]], %[[V1]], %[[V2]], %[[V3]] : vector<32xf32>, vector<32xf32>, vector<64xf32>, vector<64xf32>
    %2 = "some_def"() : () -> (vector<32xf32>)
    %3 = "some_def"() : () -> (vector<32xf32>)
    %4 = "some_def"() : () -> (vector<64xf32>)
    %5 = "some_def"() : () -> (vector<64xf32>)
    %6 = arith.addf %2, %3 : vector<32xf32>
    %7 = arith.addf %4, %5 : vector<64xf32>
    vector_ext.yield %6, %7 : vector<32xf32>, vector<64xf32>
  }
  // CHECK: %[[A0:.*]] = arith.addf %[[R]]#2, %[[R]]#3 : vector<2xf32>
  // CHECK: %[[A1:.*]] = arith.addf %[[R]]#0, %[[R]]#1 : vector<1xf32>
  %id2 = affine.apply #map0()[%laneid] 
  // CHECK: vector.transfer_write %[[A1]], {{.*}} : vector<1xf32>, memref<1024xf32>
  // CHECK: vector.transfer_write %[[A0]], {{.*}} : vector<2xf32>, memref<1024xf32>
  vector.transfer_write %r#0, %dest[%laneid] : vector<1xf32>, memref<1024xf32>
  vector.transfer_write %r#1, %dest[%id2] : vector<2xf32>, memref<1024xf32>
  return
}


// -----

#map0 = affine_map<()[s0] -> (s0 * 2)>

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 2)>

// CHECK:   func @warp_propagate_read
//  CHECK-SAME:     (%[[ID:.*]]: index
func @warp_propagate_read(%laneid: index, %src: memref<1024xf32>, %dest: memref<1024xf32>) {
// CHECK-NOT: warp_execute_on_lane_0
// CHECK: %[[R0:.*]] = vector.transfer_read %arg1[%[[ID]]], %{{.*}} : memref<1024xf32>, vector<1xf32>
// CHECK: %[[ID2:.*]] = affine.apply #[[MAP0]]()[%[[ID]]]
// CHECK: %[[R1:.*]] = vector.transfer_read %arg1[%[[ID2]]], %{{.*}} : memref<1024xf32>, vector<2xf32>
// CHECK: vector.transfer_write %[[R0]], {{.*}} : vector<1xf32>, memref<1024xf32>
// CHECK: vector.transfer_write %[[R1]], {{.*}} : vector<2xf32>, memref<1024xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r:2 = vector_ext.warp_execute_on_lane_0(%laneid) ->(vector<1xf32>, vector<2xf32>) {
    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<32xf32>
    %3 = vector.transfer_read %src[%c32], %cst : memref<1024xf32>, vector<64xf32>
    vector_ext.yield %2, %3 : vector<32xf32>, vector<64xf32>
  }
  %id2 = affine.apply #map0()[%laneid] 
  vector.transfer_write %r#0, %dest[%laneid] : vector<1xf32>, memref<1024xf32>
  vector.transfer_write %r#1, %dest[%id2] : vector<2xf32>, memref<1024xf32>
  return
}

// -----

// CHECK-SCF-IF-LABEL: func @rewrite_warp_op_to_scf_if(
//  CHECK-SCF-IF-SAME:     %[[laneid:.*]]: index,
//  CHECK-SCF-IF-SAME:     %[[v0:.*]]: vector<4xf32>, %[[v1:.*]]: vector<8xf32>)
func @rewrite_warp_op_to_scf_if(%laneid: index,
                                %v0: vector<4xf32>, %v1: vector<8xf32>) {
//   CHECK-SCF-IF-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-SCF-IF-DAG:   %[[c2:.*]] = arith.constant 2 : index
//   CHECK-SCF-IF-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-SCF-IF-DAG:   %[[c8:.*]] = arith.constant 8 : index
//   CHECK-SCF-IF-DAG:   %[[c32:.*]] = arith.constant 32 : index
//   CHECK-SCF-IF-DAG:   %[[c128:.*]] = arith.constant 128 : index
//       CHECK-SCF-IF:   %[[buffer:.*]] = memref.alloc() : memref<384xf32>
//       CHECK-SCF-IF:   %[[is_lane_0:.*]] = arith.cmpi eq, %[[laneid]], %[[c0]]

//       CHECK-SCF-IF:   %[[s0:.*]] = arith.muli %[[laneid]], %[[c4]]
//       CHECK-SCF-IF:   vector.store %[[v0]], %[[buffer]][%[[s0]]]
//       CHECK-SCF-IF:   %[[mul1:.*]] = arith.muli %[[laneid]], %[[c8]]
//       CHECK-SCF-IF:   %[[s1:.*]] = arith.addi %[[mul1]], %[[c128]]
//       CHECK-SCF-IF:   vector.store %[[v1]], %[[buffer]][%[[s1]]]

//       CHECK-SCF-IF:   scf.if %[[is_lane_0]] {
  %r:2 = vector_ext.warp_execute_on_lane_0(%laneid)
      args(%v0, %v1 : vector<4xf32>, vector<8xf32>) -> (vector<1xf32>, vector<2xf32>) {
    ^bb0(%arg0: vector<128xf32>, %arg1: vector<256xf32>):
//       CHECK-SCF-IF:     %[[arg1:.*]] = vector.load %[[buffer]][%[[c128]]] : memref<384xf32>, vector<256xf32>
//       CHECK-SCF-IF:     %[[arg0:.*]] = vector.load %[[buffer]][%[[c0]]] : memref<384xf32>, vector<128xf32>
//       CHECK-SCF-IF:     %[[def_0:.*]] = "some_def"(%[[arg0]]) : (vector<128xf32>) -> vector<32xf32>
//       CHECK-SCF-IF:     %[[def_1:.*]] = "some_def"(%[[arg1]]) : (vector<256xf32>) -> vector<64xf32>
    %2 = "some_def"(%arg0) : (vector<128xf32>) -> vector<32xf32>
    %3 = "some_def"(%arg1) : (vector<256xf32>) -> vector<64xf32>
//       CHECK-SCF-IF:     vector.store %[[def_0]], %[[buffer]][%[[c0]]]
//       CHECK-SCF-IF:     vector.store %[[def_1]], %[[buffer]][%[[c32]]]
    vector_ext.yield %2, %3 : vector<32xf32>, vector<64xf32>
  }
//       CHECK-SCF-IF:   }
//       CHECK-SCF-IF:   %[[o1:.*]] = arith.muli %[[laneid]], %[[c2]]
//       CHECK-SCF-IF:   %[[o2:.*]] = arith.addi %[[o1]], %[[c32]]
//       CHECK-SCF-IF:   %[[r1:.*]] = vector.load %[[buffer]][%[[o2]]] : memref<384xf32>, vector<2xf32>
//       CHECK-SCF-IF:   %[[r0:.*]] = vector.load %[[buffer]][%[[laneid]]] : memref<384xf32>, vector<1xf32>
//       CHECK-SCF-IF:   vector.print %[[r0]]
//       CHECK-SCF-IF:   vector.print %[[r1]]
  vector.print %r#0 : vector<1xf32>
  vector.print %r#1 : vector<2xf32>
  return
}