// RUN: iterators-opt %s | iterators-opt | FileCheck %s

// CHECK-LABEL: func.func @to_elements_i32(
// CHECK-SAME:                             %[[ARG0:.*]]: tuple<i32>) -> i32 {
// CHECK-NEXT:    %[[V0:elements.*]] = tuple.to_elements %[[ARG0]] : tuple<i32>
// CHECK-NEXT:    return %[[V0]] : i32
func.func @to_elements_i32(%arg0 : tuple<i32>) -> i32 {
  %elements = tuple.to_elements %arg0 : tuple<i32>
  return %elements : i32
}

// CHECK-LABEL: func.func @to_elements_empty(
// CHECK-SAME:                               %[[ARG0:.*]]: tuple<>) {
// CHECK-NEXT:    tuple.to_elements %[[ARG0]] : tuple<>
// CHECK-NEXT:    return
func.func @to_elements_empty(%arg0 : tuple<>) {
  tuple.to_elements %arg0 : tuple<>
  return
}
