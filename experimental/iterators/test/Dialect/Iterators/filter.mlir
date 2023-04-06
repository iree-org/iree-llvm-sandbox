// RUN: iterators-opt %s \
// RUN: | FileCheck %s


func.func private @is_positive_tuple(%tuple : tuple<i32>) -> i1 {
  %i = tuple.to_elements %tuple : tuple<i32>
  %zero = arith.constant 0 : i32
  %cmp = arith.cmpi "sgt", %i, %zero : i32
  return %cmp : i1
}

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %filtered = "iterators.filter"(%input) {predicateRef = @is_positive_tuple} :
                  (!iterators.stream<tuple<i32>>) ->
                      (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V1:filtered.*]] = "iterators.filter"(%[[V0]]) {predicateRef = @is_positive_tuple} : (!iterators.stream<tuple<i32>>) -> !iterators.stream<tuple<i32>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
