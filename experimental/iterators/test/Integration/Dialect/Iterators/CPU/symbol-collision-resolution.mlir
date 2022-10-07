// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!element_type = !llvm.struct<(i32)>

func.func private @sum_struct(%lhs : !element_type, %rhs : !element_type) -> !element_type {
  %lhsi = llvm.extractvalue %lhs[0 : index] : !element_type
  %rhsi = llvm.extractvalue %rhs[0 : index] : !element_type
  %i = arith.addi %lhsi, %rhsi : i32
  %result = llvm.insertvalue %i, %lhs[0 : index] : !element_type
  return %result : !element_type
}

func.func @query1() {
  %input = "iterators.constantstream"() { value = [[6 : i32]] }
      : () -> (!iterators.stream<!element_type>)
  %reduce1 = "iterators.reduce"(%input) {reduceFuncRef = @sum_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  // Reduce result again to vary from second query.
  %reduce2 = "iterators.reduce"(%reduce1) {reduceFuncRef = @sum_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%reduce2) : (!iterators.stream<!element_type>) -> ()
  return
}

func.func @query2() {
  // Run similar query again to check that name collision resolution works.
  %input = "iterators.constantstream"(){ value = [[6 : i32]] }
      : () -> (!iterators.stream<!element_type>)
  %reduce = "iterators.reduce"(%input) {reduceFuncRef = @sum_struct}
    : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%reduce) : (!iterators.stream<!element_type>) -> ()
  return
}
func.func @main() {
  // CHECK: (6)
  call @query1() : () -> ()

  // CHECK: (6)
  call @query2() : () -> ()
  return
}
