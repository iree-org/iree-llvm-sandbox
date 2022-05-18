// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!tupleType = type !llvm.struct<(i32)>

func @query1() {
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<!tupleType>)
  %reduce1 = "iterators.reduce"(%input) : (!iterators.stream<!tupleType>) -> (!iterators.stream<!tupleType>)
  // Reduce result again to vary from second query.
  %reduce2 = "iterators.reduce"(%reduce1) : (!iterators.stream<!tupleType>) -> (!iterators.stream<!tupleType>)
  "iterators.sink"(%reduce2) : (!iterators.stream<!tupleType>) -> ()
  return
}

func @query2() {
  // Run similar query again to check that name collision resolution works.
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<!tupleType>)
  %reduce = "iterators.reduce"(%input) : (!iterators.stream<!tupleType>) -> (!iterators.stream<!tupleType>)
  "iterators.sink"(%reduce) : (!iterators.stream<!tupleType>) -> ()
  return
}
func @main() {
  // CHECK: (6)
  call @query1() : () -> ()

  // CHECK: (6)
  call @query2() : () -> ()
  return
}
