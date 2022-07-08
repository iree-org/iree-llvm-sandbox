// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!element_type = !llvm.struct<(i32)>

func.func private @is_positive_struct(%struct : !element_type) -> i1 {
  %i = llvm.extractvalue %struct[0 : index] : !element_type
  %zero = arith.constant 0 : i32
  %cmp = arith.cmpi "sgt", %i, %zero : i32
  return %cmp : i1
}

func.func @main() {
  %input = "iterators.constantstream"()
    { value = [[0: i32], [1: i32], [-1: i32], [2: i32], [-2: i32]] }
    : () -> (!iterators.stream<!element_type>)
  %filter = "iterators.filter"(%input) {predicateRef = @is_positive_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%filter) : (!iterators.stream<!element_type>) -> ()
  // CHECK:      (1)
  // CHECK-NEXT: (2)
  return
}
