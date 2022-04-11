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
  } {warp_size = 32}
  // CHECK: return %[[R]] : vector<1xf32>
  return %r#1 : vector<1xf32>
}

// -----

// CHECK-LABEL:   func @warp_propagate_operand(
//  CHECK-SAME:   %[[ID:.*]]: index, %[[V:.*]]: vector<4xf32>)
func @warp_propagate_operand(%laneid: index, %v0: vector<4xf32>)
  -> (vector<4xf32>) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid)
     args(%v0 : vector<4xf32>) -> (vector<4xf32>) {
     ^bb0(%arg0 : vector<128xf32>) :
    vector_ext.yield %arg0 : vector<128xf32>
  } {warp_size = 32}
  // CHECK: return %[[V]] : vector<4xf32>
  return %r : vector<4xf32>
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
  } {warp_size = 32}
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

// CHECK-LABEL: func @warp_propagate_scalar_arith(
//       CHECK:   %[[r:.*]]:2 = vector_ext.warp_execute_on_lane_0{{.*}} {
//       CHECK:     %[[some_def0:.*]] = "some_def"
//       CHECK:     %[[some_def1:.*]] = "some_def"
//       CHECK:     vector_ext.yield %[[some_def0]], %[[some_def1]]
//       CHECK:   }
//       CHECK:   arith.addf %[[r]]#0, %[[r]]#1 : f32
func @warp_propagate_scalar_arith(%laneid: index) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (f32) {
    %0 = "some_def"() : () -> (f32)
    %1 = "some_def"() : () -> (f32)
    %2 = arith.addf %0, %1 : f32
    vector_ext.yield %2 : f32
  } {warp_size = 32}
  vector.print %r : f32
  return
}

// -----

// CHECK-LABEL: func @warp_propagate_cast(
//   CHECK-NOT:   vector_ext.warp_execute_on_lane_0
//       CHECK:   %[[result:.*]] = arith.sitofp %{{.*}} : i32 to f32
//       CHECK:   return %[[result]]
func @warp_propagate_cast(%laneid : index, %i : i32) -> (f32) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (f32) {
    %casted = arith.sitofp %i : i32 to f32
    vector_ext.yield %casted : f32
  } {warp_size = 32}
  return %r : f32
}

// -----

#map0 = affine_map<()[s0] -> (s0 * 2)>

//  CHECK-DAG: #[[MAP0:.*]] = affine_map<()[s0] -> (s0 * 2)>

// CHECK:   func @warp_propagate_read
//  CHECK-SAME:     (%[[ID:.*]]: index
func @warp_propagate_read(%laneid: index, %src: memref<1024xf32>, %dest: memref<1024xf32>) {
// CHECK-NOT: warp_execute_on_lane_0
// CHECK-DAG: %[[R0:.*]] = vector.transfer_read %arg1[%[[ID]]], %{{.*}} : memref<1024xf32>, vector<1xf32>
// CHECK-DAG: %[[ID2:.*]] = affine.apply #[[MAP0]]()[%[[ID]]]
// CHECK-DAG: %[[R1:.*]] = vector.transfer_read %arg1[%[[ID2]]], %{{.*}} : memref<1024xf32>, vector<2xf32>
// CHECK: vector.transfer_write %[[R0]], {{.*}} : vector<1xf32>, memref<1024xf32>
// CHECK: vector.transfer_write %[[R1]], {{.*}} : vector<2xf32>, memref<1024xf32>
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %r:2 = vector_ext.warp_execute_on_lane_0(%laneid) ->(vector<1xf32>, vector<2xf32>) {
    %2 = vector.transfer_read %src[%c0], %cst : memref<1024xf32>, vector<32xf32>
    %3 = vector.transfer_read %src[%c32], %cst : memref<1024xf32>, vector<64xf32>
    vector_ext.yield %2, %3 : vector<32xf32>, vector<64xf32>
  } {warp_size = 32}
  %id2 = affine.apply #map0()[%laneid]
  vector.transfer_write %r#0, %dest[%laneid] : vector<1xf32>, memref<1024xf32>
  vector.transfer_write %r#1, %dest[%id2] : vector<2xf32>, memref<1024xf32>
  return
}

// -----

// CHECK-SCF-IF-DAG: memref.global "private" @__shared_32xf32 : memref<32xf32, 3>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_64xf32 : memref<64xf32, 3>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_128xf32 : memref<128xf32, 3>
// CHECK-SCF-IF-DAG: memref.global "private" @__shared_256xf32 : memref<256xf32, 3>

// CHECK-SCF-IF-LABEL: func @rewrite_warp_op_to_scf_if(
//  CHECK-SCF-IF-SAME:     %[[laneid:.*]]: index,
//  CHECK-SCF-IF-SAME:     %[[v0:.*]]: vector<4xf32>, %[[v1:.*]]: vector<8xf32>)
func @rewrite_warp_op_to_scf_if(%laneid: index,
                                %v0: vector<4xf32>, %v1: vector<8xf32>) {
//   CHECK-SCF-IF-DAG:   %[[c0:.*]] = arith.constant 0 : index
//   CHECK-SCF-IF-DAG:   %[[c2:.*]] = arith.constant 2 : index
//   CHECK-SCF-IF-DAG:   %[[c4:.*]] = arith.constant 4 : index
//   CHECK-SCF-IF-DAG:   %[[c8:.*]] = arith.constant 8 : index
//       CHECK-SCF-IF:   %[[is_lane_0:.*]] = arith.cmpi eq, %[[laneid]], %[[c0]]

//       CHECK-SCF-IF:   %[[buffer_v0:.*]] = memref.get_global @__shared_128xf32
//       CHECK-SCF-IF:   %[[s0:.*]] = arith.muli %[[laneid]], %[[c4]]
//       CHECK-SCF-IF:   vector.store %[[v0]], %[[buffer_v0]][%[[s0]]]
//       CHECK-SCF-IF:   %[[buffer_v1:.*]] = memref.get_global @__shared_256xf32
//       CHECK-SCF-IF:   %[[s1:.*]] = arith.muli %[[laneid]], %[[c8]]
//       CHECK-SCF-IF:   vector.store %[[v1]], %[[buffer_v1]][%[[s1]]]

//       CHECK-SCF-IF:   %[[buffer_def_0:.*]] = memref.get_global @__shared_32xf32
//       CHECK-SCF-IF:   %[[buffer_def_1:.*]] = memref.get_global @__shared_64xf32

//       CHECK-SCF-IF:   scf.if %[[is_lane_0]] {
  %r:2 = vector_ext.warp_execute_on_lane_0(%laneid)
      args(%v0, %v1 : vector<4xf32>, vector<8xf32>) -> (vector<1xf32>, vector<2xf32>) {
    ^bb0(%arg0: vector<128xf32>, %arg1: vector<256xf32>):
//       CHECK-SCF-IF:     %[[arg1:.*]] = vector.load %[[buffer_v1]][%[[c0]]] : memref<256xf32, 3>, vector<256xf32>
//       CHECK-SCF-IF:     %[[arg0:.*]] = vector.load %[[buffer_v0]][%[[c0]]] : memref<128xf32, 3>, vector<128xf32>
//       CHECK-SCF-IF:     %[[def_0:.*]] = "some_def"(%[[arg0]]) : (vector<128xf32>) -> vector<32xf32>
//       CHECK-SCF-IF:     %[[def_1:.*]] = "some_def"(%[[arg1]]) : (vector<256xf32>) -> vector<64xf32>
    %2 = "some_def"(%arg0) : (vector<128xf32>) -> vector<32xf32>
    %3 = "some_def"(%arg1) : (vector<256xf32>) -> vector<64xf32>
//       CHECK-SCF-IF:     vector.store %[[def_0]], %[[buffer_def_0]][%[[c0]]]
//       CHECK-SCF-IF:     vector.store %[[def_1]], %[[buffer_def_1]][%[[c0]]]
    vector_ext.yield %2, %3 : vector<32xf32>, vector<64xf32>
  } {warp_size = 32}
//       CHECK-SCF-IF:   }
//       CHECK-SCF-IF:   %[[o1:.*]] = arith.muli %[[laneid]], %[[c2]]
//       CHECK-SCF-IF:   %[[r1:.*]] = vector.load %[[buffer_def_1]][%[[o1]]] : memref<64xf32, 3>, vector<2xf32>
//       CHECK-SCF-IF:   %[[r0:.*]] = vector.load %[[buffer_def_0]][%[[laneid]]] : memref<32xf32, 3>, vector<1xf32>
//       CHECK-SCF-IF:   vector.print %[[r0]]
//       CHECK-SCF-IF:   vector.print %[[r1]]
  vector.print %r#0 : vector<1xf32>
  vector.print %r#1 : vector<2xf32>
  return
}

// -----

// CHECK-SCF-IF-DAG: memref.global "private" @__shared_1xf32 : memref<1xf32, 3>

// CHECK-SCF-IF-LABEL: func @vector_reduction(
//  CHECK-SCF-IF-SAME:     %[[laneid:.*]]: index)
//       CHECK-SCF-IF:   %[[c0:.*]] = arith.constant 0 : index
//       CHECK-SCF-IF:   %[[is_lane_0:.*]] = arith.cmpi eq, %[[laneid]]
//       CHECK-SCF-IF:   %[[buffer:.*]] = memref.get_global @__shared_1xf32 : memref<1xf32, 3>
//       CHECK-SCF-IF:   scf.if %[[is_lane_0]] {
//       CHECK-SCF-IF:     %[[reduction:.*]] = vector.reduction
//       CHECK-SCF-IF:     memref.store %[[reduction]], %[[buffer]][%[[c0]]]
//       CHECK-SCF-IF:   }
//       CHECK-SCF-IF:   %[[broadcasted:.*]] = memref.load %[[buffer]][%[[c0]]]
//       CHECK-SCF-IF:   vector.print %[[broadcasted]] : f32

// CHECK-LABEL: func @vector_reduction(
//  CHECK-SAME:     %[[laneid:.*]]: index)
//   CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : i32
//   CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : i32
//   CHECK-DAG:   %[[c2:.*]] = arith.constant 2 : i32
//   CHECK-DAG:   %[[c4:.*]] = arith.constant 4 : i32
//   CHECK-DAG:   %[[c8:.*]] = arith.constant 8 : i32
//   CHECK-DAG:   %[[c16:.*]] = arith.constant 16 : i32
//   CHECK-DAG:   %[[c32:.*]] = arith.constant 32 : i32
//       CHECK:   %[[warp_op:.*]] = vector_ext.warp_execute_on_lane_0(%[[laneid]]) -> (vector<1xf32>) {
//       CHECK:     vector_ext.yield %{{.*}} : vector<32xf32>
//       CHECK:   }
//       CHECK:   %[[a:.*]] = vector.extract %[[warp_op]][0] : vector<1xf32>
//       CHECK:   %[[r0:.*]], %{{.*}} = gpu.shuffle  down %[[a]], %[[c16]], %[[c32]]
//       CHECK:   %[[a0:.*]] = arith.addf %[[a]], %[[r0]]
//       CHECK:   %[[r1:.*]], %{{.*}} = gpu.shuffle  down %[[a0]], %[[c8]], %[[c32]]
//       CHECK:   %[[a1:.*]] = arith.addf %[[a0]], %[[r1]]
//       CHECK:   %[[r2:.*]], %{{.*}} = gpu.shuffle  down %[[a1]], %[[c4]], %[[c32]]
//       CHECK:   %[[a2:.*]] = arith.addf %[[a1]], %[[r2]]
//       CHECK:   %[[r3:.*]], %{{.*}} = gpu.shuffle  down %[[a2]], %[[c2]], %[[c32]]
//       CHECK:   %[[a3:.*]] = arith.addf %[[a2]], %[[r3]]
//       CHECK:   %[[r4:.*]], %{{.*}} = gpu.shuffle  down %[[a3]], %[[c1]], %[[c32]]
//       CHECK:   %[[a4:.*]] = arith.addf %[[a3]], %[[r4]]
//       CHECK:   %[[broadcasted:.*]], %{{.*}} = gpu.shuffle  idx %[[a4]], %[[c0]], %[[c32]]
//       CHECK:   vector.print %[[broadcasted]] : f32
func @vector_reduction(%laneid: index) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (f32) {
    %0 = "some_def"() : () -> (vector<32xf32>)
    %1 = vector.reduction <add>, %0 : vector<32xf32> into f32
    vector_ext.yield %1 : f32
  } {warp_size = 32}
  vector.print %r : f32
  return
}

// -----

// CHECK-LABEL: func @fold_vector_broadcast(
//       CHECK:   %[[r:.*]] = vector_ext.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK:     %[[some_def:.*]] = "some_def"
//       CHECK:     vector_ext.yield %[[some_def]] : vector<1xf32>
//       CHECK:   vector.print %[[r]] : vector<1xf32>
func @fold_vector_broadcast(%laneid: index) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (vector<1xf32>) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %1 = vector.broadcast %0 : vector<1xf32> to vector<32xf32>
    vector_ext.yield %1 : vector<32xf32>
  } {warp_size = 32}
  vector.print %r : vector<1xf32>
  return
}

// -----

// CHECK-LABEL: func @extract_vector_broadcast(
//       CHECK:   %[[r:.*]] = vector_ext.warp_execute_on_lane_0{{.*}} -> (vector<1xf32>)
//       CHECK:     %[[some_def:.*]] = "some_def"
//       CHECK:     vector_ext.yield %[[some_def]] : vector<1xf32>
//       CHECK:   %[[broadcasted:.*]] = vector.broadcast %[[r]] : vector<1xf32> to vector<2xf32>
//       CHECK:   vector.print %[[broadcasted]] : vector<2xf32>
func @extract_vector_broadcast(%laneid: index) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (vector<1xf32>)
    %1 = vector.broadcast %0 : vector<1xf32> to vector<64xf32>
    vector_ext.yield %1 : vector<64xf32>
  } {warp_size = 32}
  vector.print %r : vector<2xf32>
  return
}

// -----

// CHECK-LABEL: func @extract_scalar_vector_broadcast(
//       CHECK:   %[[r:.*]] = vector_ext.warp_execute_on_lane_0{{.*}} -> (f32)
//       CHECK:     %[[some_def:.*]] = "some_def"
//       CHECK:     vector_ext.yield %[[some_def]] : f32
//       CHECK:   %[[broadcasted:.*]] = vector.broadcast %[[r]] : f32 to vector<2xf32>
//       CHECK:   vector.print %[[broadcasted]] : vector<2xf32>
func @extract_scalar_vector_broadcast(%laneid: index) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (vector<2xf32>) {
    %0 = "some_def"() : () -> (f32)
    %1 = vector.broadcast %0 : f32 to vector<64xf32>
    vector_ext.yield %1 : vector<64xf32>
  } {warp_size = 32}
  vector.print %r : vector<2xf32>
  return
}

// -----

// CHECK-LABEL:   func @warp_scf_for(
// CHECK: %[[INI:.*]] = vector_ext.warp_execute_on_lane_0(%{{.*}}) -> (vector<4xf32>) {
// CHECK:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK:   vector_ext.yield %[[INI1]] : vector<128xf32>
// CHECK: } {warp_size = 32 : i64}
// CHECK: %[[F:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG:.*]] = %[[INI]]) -> (vector<4xf32>) {
// CHECK:   %[[W:.*]] = vector_ext.warp_execute_on_lane_0(%{{.*}}) args(%[[FARG]] : vector<4xf32>) -> (vector<4xf32>) {
// CHECK:    ^bb0(%[[ARG:.*]]: vector<128xf32>):
// CHECK:      %[[ACC:.*]] = "some_def"(%[[ARG]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK:      vector_ext.yield %[[ACC]] : vector<128xf32>
// CHECK:   } {warp_size = 32 : i64}
// CHECK:   scf.yield %[[W]] : vector<4xf32>
// CHECK: }
// CHECK: "some_use"(%[[F]]) : (vector<4xf32>) -> ()
func @warp_scf_for(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<4xf32>) {
    %ini = "some_def"() : () -> (vector<128xf32>)
    %3 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini) -> (vector<128xf32>) {
      %acc = "some_def"(%arg4) : (vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc : vector<128xf32>
    }
    vector_ext.yield %3 : vector<128xf32>
  } {warp_size = 32}
  "some_use"(%0) : (vector<4xf32>) -> ()
  return
}

// -----

// CHECK-LABEL:   func @warp_scf_for_swap(
// CHECK: %[[INI:.*]]:2 = vector_ext.warp_execute_on_lane_0(%{{.*}}) -> (vector<4xf32>, vector<4xf32>) {
// CHECK:   %[[INI1:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK:   %[[INI2:.*]] = "some_def"() : () -> vector<128xf32>
// CHECK:   vector_ext.yield %[[INI1]], %[[INI2]] : vector<128xf32>, vector<128xf32>
// CHECK: } {warp_size = 32 : i64}
// CHECK: %[[F:.*]]:2 = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[FARG1:.*]] = %[[INI]]#0, %[[FARG2:.*]] = %[[INI]]#1) -> (vector<4xf32>, vector<4xf32>) {
// CHECK:   %[[W:.*]]:2 = vector_ext.warp_execute_on_lane_0(%{{.*}}) args(%[[FARG1]], %[[FARG2]] : vector<4xf32>, vector<4xf32>) -> (vector<4xf32>, vector<4xf32>) {
// CHECK:    ^bb0(%[[ARG1:.*]]: vector<128xf32>, %[[ARG2:.*]]: vector<128xf32>):
// CHECK:      %[[ACC1:.*]] = "some_def"(%[[ARG1]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK:      %[[ACC2:.*]] = "some_def"(%[[ARG2]]) : (vector<128xf32>) -> vector<128xf32>
// CHECK:      vector_ext.yield %[[ACC2]], %[[ACC1]] : vector<128xf32>, vector<128xf32>
// CHECK:   } {warp_size = 32 : i64}
// CHECK:   scf.yield %[[W]]#0, %[[W]]#1 : vector<4xf32>, vector<4xf32>
// CHECK: }
// CHECK: "some_use"(%[[F]]#0) : (vector<4xf32>) -> ()
// CHECK: "some_use"(%[[F]]#1) : (vector<4xf32>) -> ()
func @warp_scf_for_swap(%arg0: index) {
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %0:2 = vector_ext.warp_execute_on_lane_0(%arg0) -> (vector<4xf32>, vector<4xf32>) {
    %ini1 = "some_def"() : () -> (vector<128xf32>)
    %ini2 = "some_def"() : () -> (vector<128xf32>)
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %ini1, %arg5 = %ini2) -> (vector<128xf32>, vector<128xf32>) {
      %acc1 = "some_def"(%arg4) : (vector<128xf32>) -> (vector<128xf32>)
      %acc2 = "some_def"(%arg5) : (vector<128xf32>) -> (vector<128xf32>)
      scf.yield %acc2, %acc1 : vector<128xf32>, vector<128xf32>
    }
    vector_ext.yield %3#0, %3#1 : vector<128xf32>, vector<128xf32>
  } {warp_size = 32}
  "some_use"(%0#0) : (vector<4xf32>) -> ()
  "some_use"(%0#1) : (vector<4xf32>) -> ()
  return
}

// -----

#map = affine_map<()[s0] -> (s0 * 4)>
#map1 = affine_map<()[s0] -> (s0 * 128 + 128)>
#map2 = affine_map<()[s0] -> (s0 * 4 + 128)>

// CHECK-LABEL:   func @warp_scf_for_multiple_yield(
//       CHECK:   vector_ext.warp_execute_on_lane_0(%{{.*}}) -> (vector<1xf32>) {
//  CHECK-NEXT:     "some_def"() : () -> vector<32xf32>
//  CHECK-NEXT:     vector_ext.yield %{{.*}} : vector<32xf32>
//  CHECK-NEXT:   } {warp_size = 32 : i64}
//   CHECK-NOT:   vector_ext.warp_execute_on_lane_0
//       CHECK:   vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK:   vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK:   %{{.*}}:2 = scf.for {{.*}} -> (vector<4xf32>, vector<4xf32>) {
//   CHECK-NOT:     vector_ext.warp_execute_on_lane_0
//       CHECK:     vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK:     vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
//       CHECK:     arith.addf {{.*}} : vector<4xf32>
//       CHECK:     arith.addf {{.*}} : vector<4xf32>
//       CHECK:     scf.yield {{.*}} : vector<4xf32>, vector<4xf32>
//       CHECK:   }
func @warp_scf_for_multiple_yield(%arg0: index, %arg1: memref<?xf32>, %arg2: memref<?xf32>) {
  %c256 = arith.constant 256 : index
  %c128 = arith.constant 128 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0:3 = vector_ext.warp_execute_on_lane_0(%arg0) ->
  (vector<1xf32>, vector<4xf32>, vector<4xf32>) {
    %def = "some_def"() : () -> (vector<32xf32>)
    %r1 = vector.transfer_read %arg2[%c0], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
    %r2 = vector.transfer_read %arg2[%c128], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
    %3:2 = scf.for %arg3 = %c0 to %c128 step %c1 iter_args(%arg4 = %r1, %arg5 = %r2)
    -> (vector<128xf32>, vector<128xf32>) {
      %o1 = affine.apply #map1()[%arg3]
      %o2 = affine.apply #map2()[%arg3]
      %4 = vector.transfer_read %arg1[%o1], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
      %5 = vector.transfer_read %arg1[%o2], %cst {in_bounds = [true]} : memref<?xf32>, vector<128xf32>
      %6 = arith.addf %4, %arg4 : vector<128xf32>
      %7 = arith.addf %5, %arg5 : vector<128xf32>
      scf.yield %6, %7 : vector<128xf32>, vector<128xf32>
    }
    vector_ext.yield %def, %3#0, %3#1 :  vector<32xf32>, vector<128xf32>, vector<128xf32>
  } {warp_size = 32}
  %1 = affine.apply #map()[%arg0]
  vector.transfer_write %0#1, %arg2[%1] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  %2 = affine.apply #map2()[%arg0]
  vector.transfer_write %0#2, %arg2[%2] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
  "some_use"(%0#0) : (vector<1xf32>) -> ()
  return
}

// -----

// CHECK-LABEL: func @large_vector_reduction(
//       CHECK:   vector_ext.warp_execute_on_lane_0(%{{.*}}) -> (vector<1xf32>, vector<1xf32>) {
//       CHECK:     %[[some_def:.*]] = "some_def"()
//       CHECK:     %[[slice1:.*]] = vector.extract_strided_slice %[[some_def]] {offsets = [0], sizes = [32]
//       CHECK:     %[[slice2:.*]] = vector.extract_strided_slice %[[some_def]] {offsets = [32], sizes = [32]
//       CHECK:     vector_ext.yield %[[slice1]], %[[slice2]]
//       CHECK:   } {warp_size = 32 : i64}
//       CHECK:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   %[[r0:.*]], %{{.*}} = gpu.shuffle  idx
//       CHECK:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   gpu.shuffle  down
//  CHECK-NEXT:   arith.addf
//  CHECK-NEXT:   %[[r1:.*]], %{{.*}} = gpu.shuffle  idx
//       CHECK:   %[[result:.*]] = arith.addf %[[r1]], %[[r0]] : f32
//       CHECK:   vector.print %[[result]]
func @large_vector_reduction(%laneid: index) {
  %r = vector_ext.warp_execute_on_lane_0(%laneid) -> (f32) {
    %0 = "some_def"() : () -> (vector<64xf32>)
    %1 = vector.reduction <add>, %0 : vector<64xf32> into f32
    vector_ext.yield %1 : f32
  } {warp_size = 32}
  vector.print %r : f32
  return
}
