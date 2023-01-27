// RUN: iterators-opt %s -inline | FileCheck %s

// CHECK-LABEL: func.func @test_inline() -> i32 {
// CHECK-NEXT: %[[V0:.*]] = iterators.undefstate : !iterators.state<i32>
// CHECK-NEXT: %[[RES:.*]] = iterators.extractvalue %[[V0]][0] : !iterators.state<i32>
// CHECK-NEXT: return %[[RES]] : i32
func.func @test_inline() -> i32 {
  %0 = call @inner_func_inlinable() : () -> i32
  return %0 : i32
}

func.func @inner_func_inlinable() -> i32 {
  %0 = iterators.undefstate : !iterators.state<i32>
  %1 = iterators.extractvalue %0[0] : !iterators.state<i32>
  return %1 : i32
}
