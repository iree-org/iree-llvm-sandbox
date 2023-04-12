// RUN: iterators-opt %s \
// RUN: | FileCheck %s

func.func private @unpack_i32(%input : tuple<i32>) -> i32 {
  %i = tuple.to_elements %input : tuple<i32>
  return %i : i32
}

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32} :
                  (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)
// CHECK-NEXT:    %[[V1:mapped.*]] = "iterators.map"(%[[V0]]) {mapFuncRef = @unpack_i32} : (!iterators.stream<tuple<i32>>) -> !iterators.stream<i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
