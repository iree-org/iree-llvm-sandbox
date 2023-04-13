// RUN: iterators-opt %s \
// RUN: | FileCheck %s

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %empty_tuple = "iterators.constanttuple"() { values = [] } : () -> tuple<>
// CHECK-NEXT:    %[[V0:tuple.*]] = "iterators.constanttuple"() {values = []} : () -> tuple<>
  %one_field_tuple = "iterators.constanttuple"()
      { values = [1 : i32] } : () -> tuple<i32>
// CHECK-NEXT:    %[[V0:tuple.*]] = "iterators.constanttuple"() {values = [1 : i32]} : () -> tuple<i32>
  %three_field_tuple = "iterators.constanttuple"()
      { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
// CHECK-NEXT:    %[[V0:tuple.*]] = "iterators.constanttuple"() {values = [1 : i32, 2 : i32, 3 : i32]} : () -> tuple<i32, i32, i32>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
