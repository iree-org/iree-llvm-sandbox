// RUN: iterators-opt %s \
// RUN: | FileCheck %s

func.func private @accumulate_sum_tuple(
      %acc : tuple<i32>, %val : tuple<i32>) -> tuple<i32> {
  %acci = tuple.to_elements %acc : tuple<i32>
  %vali = tuple.to_elements %val : tuple<i32>
  %i = arith.addi %acci, %vali : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

// CHECK-LABEL: func.func @main() {
func.func @main() {
// CHECK-NEXT:    %[[V0:.*]] = "iterators.constantstream"{{.*}}
  %input = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<tuple<i32>>)

  // CHECK: %[[V1:.*]] = tuple.from_elements %{{.*}} : tuple<i32>
  %hundred = arith.constant 100 : i32
  %init_value = tuple.from_elements %hundred : tuple<i32>

  // CHECK: %[[V2:accumulated.*]] = iterators.accumulate(%[[V0]], %[[V1]]) with @accumulate_sum_tuple : (!iterators.stream<tuple<i32>>) -> !iterators.stream<tuple<i32>>
  %accumulated = iterators.accumulate(%input, %init_value)
                     with @accumulate_sum_tuple :
                         (!iterators.stream<tuple<i32>>) ->
                             !iterators.stream<tuple<i32>>
  return
}
