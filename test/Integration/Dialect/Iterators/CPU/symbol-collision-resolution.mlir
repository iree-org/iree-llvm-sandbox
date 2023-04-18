// RUN: structured-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -decompose-tuples \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func private @sum_tuple(%lhs : tuple<i32>, %rhs : tuple<i32>) -> tuple<i32> {
  %lhsi = tuple.to_elements %lhs : tuple<i32>
  %rhsi = tuple.to_elements %rhs : tuple<i32>
  %i = arith.addi %lhsi, %rhsi : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

func.func @query1() {
  // CHECK-LABEL: query1
  // CHECK-NEXT:  (6)
  // CHECK-NEXT:  -
  iterators.print("query1")
  %input = "iterators.constantstream"() { value = [[6 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %reduce1 = "iterators.reduce"(%input) {reduceFuncRef = @sum_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  // Reduce result again to vary from second query.
  %reduce2 = "iterators.reduce"(%reduce1) {reduceFuncRef = @sum_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  "iterators.sink"(%reduce2) : (!iterators.stream<tuple<i32>>) -> ()
  return
}

func.func @query2() {
  iterators.print("query2")
  // CHECK-LABEL: query2
  // CHECK-NEXT:  (6)
  // CHECK-NEXT:  -

  // Run similar query again to check that name collision resolution works.
  %input = "iterators.constantstream"(){ value = [[6 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %reduce = "iterators.reduce"(%input) {reduceFuncRef = @sum_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  "iterators.sink"(%reduce) : (!iterators.stream<tuple<i32>>) -> ()
  return
}
func.func @main() {
  call @query1() : () -> ()
  call @query2() : () -> ()
  return
}
