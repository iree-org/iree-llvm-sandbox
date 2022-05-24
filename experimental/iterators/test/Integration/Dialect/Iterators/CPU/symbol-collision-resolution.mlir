// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!element_type = type !llvm.struct<(i32)>

func @query1() {
  %input = "iterators.constantstream"() { value = [[6 : i32]] }
      : () -> (!iterators.stream<!element_type>)
  %reduce1 = "iterators.reduce"(%input)
      : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  // Reduce result again to vary from second query.
  %reduce2 = "iterators.reduce"(%reduce1)
      : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%reduce2) : (!iterators.stream<!element_type>) -> ()
  return
}

func @query2() {
  // Run similar query again to check that name collision resolution works.
  %input = "iterators.constantstream"(){ value = [[6 : i32]] }
      : () -> (!iterators.stream<!element_type>)
  %reduce = "iterators.reduce"(%input)
      : (!iterators.stream<!element_type>) -> (!iterators.stream<!element_type>)
  "iterators.sink"(%reduce) : (!iterators.stream<!element_type>) -> ()
  return
}
func @main() {
  // CHECK: (6)
  call @query1() : () -> ()

  // CHECK: (6)
  call @query2() : () -> ()
  return
}
