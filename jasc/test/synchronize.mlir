// RUN: jasc-opt %s -jasc-apply-transform-script \
// RUN:   | FileCheck %s


// CHECK-LABEL: @synchronize
func.func @synchronize(%arg0 : i32, %arg1 : i32, %arg2 : i32) {
  // Check that a gpu.barrier is inserted after the loop
  // CHECK: scf.for
  // CHECK: }
  // CHECK: gpu.barrier
  // CHECK: return
  scf.for %i0 = %arg0 to %arg1 step %arg2 : i32 {
  }
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["scf.for"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %1 = transform.jasc.synchronize %0 : (!transform.any_op) -> (!transform.op<"gpu.barrier">)
}
