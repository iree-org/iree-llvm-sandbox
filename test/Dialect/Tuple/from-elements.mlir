// RUN: iterators-opt %s | iterators-opt | FileCheck %s

// CHECK-LABEL: func.func @from_elements_i32(
// CHECK-SAME:                               %[[ARG0:.*]]: i32) -> tuple<i32> {
// CHECK-NEXT:    %[[V0:tuple.*]] = tuple.from_elements %[[ARG0]] : tuple<i32>
// CHECK-NEXT:    return %[[V0]] : tuple<i32>
func.func @from_elements_i32(%arg0 : i32) -> tuple<i32> {
  %tuple = tuple.from_elements %arg0 : tuple<i32>
  return %tuple : tuple<i32>
}

// CHECK-LABEL: func.func @from_elements_empty() -> tuple<> {
// CHECK-NEXT:    %[[V0:tuple.*]] = tuple.from_elements : tuple<>
// CHECK-NEXT:    return %[[V0]] : tuple<>
func.func @from_elements_empty() -> tuple<> {
  %tuple = tuple.from_elements : tuple<>
  return %tuple : tuple<>
}
