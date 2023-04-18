// RUN: structured-opt %s \
// RUN: | FileCheck %s

func.func private @sum_tuple(%lhs : tuple<i32>, %rhs : tuple<i32>) -> tuple<i32> {
  %lhsi = tuple.to_elements %lhs : tuple<i32>
  %rhsi = tuple.to_elements %rhs : tuple<i32>
  %i = arith.addi %lhsi, %rhsi : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %reduced = "iterators.reduce"(%input) {reduceFuncRef = @sum_tuple} :
                 (!iterators.stream<tuple<i32>>) ->
                     (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V1:reduced.*]] = "iterators.reduce"(%[[V0]]) {reduceFuncRef = @sum_tuple} : (!iterators.stream<tuple<i32>>) -> !iterators.stream<tuple<i32>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
