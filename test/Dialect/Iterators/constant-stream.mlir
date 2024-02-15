// RUN: structured-opt %s \
// RUN: | FileCheck %s

func.func @main() {
// CHECK-LABEL: func.func @main() {
  %empty = "iterators.constantstream"() { value = [] } :
               () -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V0:constantstream.*]] = "iterators.constantstream"() <{value = []}> : () -> !iterators.stream<tuple<i32>>
  %i32 = "iterators.constantstream"() { value = [[42 : i32]] } :
             () -> (!iterators.stream<tuple<i32>>)
// CHECK-NEXT:    %[[V1:constantstream.*]] = "iterators.constantstream"() <{value = {{\[}}[42 : i32]]}> : () -> !iterators.stream<tuple<i32>>
  %f32 = "iterators.constantstream"() { value = [[42. : f32]] } :
             () -> (!iterators.stream<tuple<f32>>)
// CHECK-NEXT:    %[[V2:constantstream.*]] = "iterators.constantstream"() <{value = {{\[}}[4.200000e+01 : f32]]}> : () -> !iterators.stream<tuple<f32>>
  %i32i64 = "iterators.constantstream"() { value = [[42 : i32, 1337 : i64]] } :
                () -> (!iterators.stream<tuple<i32, i64>>)
// CHECK-NEXT:    %[[V3:constantstream.*]] = "iterators.constantstream"() <{value = {{\[}}[42 : i32, 1337]]}> : () -> !iterators.stream<tuple<i32, i64>>
  return
// CHECK-NEXT:    return
}
// CHECK-NEXT:  }
