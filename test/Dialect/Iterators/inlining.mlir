// RUN: structured-opt %s -inline | FileCheck %s

// CHECK-LABEL: func.func @test_inline() -> i32 {
// CHECK-NEXT: %[[V0:.*]] = arith.constant 0 : i32
// CHECK-NEXT: %[[V1:.*]] = iterators.createstate(%[[V0]]) : !iterators.state<i32>
// CHECK-NEXT: %[[RES:.*]] = iterators.extractvalue %[[V1]][0] : !iterators.state<i32>
// CHECK-NEXT: return %[[RES]] : i32
func.func @test_inline() -> i32 {
  %0 = call @inner_func_inlinable() : () -> i32
  return %0 : i32
}

func.func @inner_func_inlinable() -> i32 {
  %0 = arith.constant 0 : i32
  %1 = iterators.createstate(%0) : !iterators.state<i32>
  %2 = iterators.extractvalue %1[0] : !iterators.state<i32>
  return %2 : i32
}
