// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!element_type = type !llvm.struct<(i32)>

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<!element_type>)
  %reduce = "iterators.reduce"(%input) : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%reduce) : (!iterators.stream<!element_type>) -> ()
  // CHECK:      (6)
  return
}
