// RUN: structured-opt %s \
// RUN:   -convert-triton-spmd-to-func-args \
// RUN: | FileCheck %s

// CHECK-LABEL: func.func public @kernel(
// CHECK-SAME:      %[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32, %[[ARG2:.*]]: i32, %[[ARG3:.*]]: i32, %[[ARG4:.*]]: i32, %[[ARG5:.*]]: i32,
// CHECK-SAME:      %[[ARG6:.*]]: !tt.ptr<i32> {tt.divisibility = 16 : i32}
// CHECK-SAME:      %[[ARG7:.*]]: i32) -> i32 {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 42 : i32
// CHECK-NEXT:    return %[[V0]] : i32
func.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: i32) -> i32 {
  %0 = arith.constant 42 : i32
  return %0 : i32
}
