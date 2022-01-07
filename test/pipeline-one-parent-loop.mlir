// RUN: mlir-proto-opt %s -outline-one-parent-loop="anchor-func=test anchor-op=scf.yield parent-loop-num=1 pipeline-factor=2" | \
// RUN: FileCheck %s

// CHECK-LABEL: func @test
func @test(%ub: index, %it: index) -> index {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %res = scf.for %i = %c0 to %ub step %c1 iter_args(%bbit = %it) -> (index) {
    scf.yield %bbit : index
  }
  return %res: index
}
