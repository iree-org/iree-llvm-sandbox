// RUN: structured-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   llvm.func @printf(!llvm.ptr, ...) -> i32
// CHECK-NEXT:   llvm.mlir.global internal constant @iterators.frmt_spec{{(\.[0-9]+)?}}("(%llu)\0A\00")
// CHECK-NEXT:   func.func @main() {
func.func @main() {
  %empty_tuple = "iterators.constanttuple"() { values = [] } : () -> tuple<>
  // CHECK-NEXT:     %[[V0:.*]] = tuple.from_elements  : tuple<>

  %one_field_tuple = "iterators.constanttuple"() { values = [1 : i32] } : () -> tuple<i32>
  // CHECK-NEXT:     %[[V2:.*]] = arith.constant 1 : i32
  // CHECK-NEXT:     %[[V3:.*]] = tuple.from_elements %[[V2]] : tuple<i32>

  iterators.print(%one_field_tuple) : tuple<i32>
  // CHECK-DAG:      %[[V4:.*]] = llvm.mlir.addressof @iterators.frmt_spec{{(\.[0-9]+)?}} : !llvm.ptr
  // CHECK-DAG:      %[[V6:.*]] = llvm.getelementptr %[[V4]][0] : (!llvm.ptr) -> !llvm.ptr, i8
  // CHECK-DAG:      %[[V7:.*]] = tuple.to_elements %[[V3]] : tuple<i32>
  // CHECK-DAG:      %[[Vb:.*]] = arith.extui %[[V7]] : i32 to i64
  // CHECK-NEXT:     %[[V8:.*]] = llvm.call @printf(%[[V6]], %[[Vb]]) : (!llvm.ptr, i64) -> i32

  %three_field_tuple = "iterators.constanttuple"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  // CHECK-DAG:      %[[V5:.*]] = arith.constant 1 : i32
  // CHECK-DAG:      %[[V1:.*]] = arith.constant 2 : i32
  // CHECK-DAG:      %[[V9:.*]] = arith.constant 3 : i32
  // CHECK-NEXT:     %[[Va:.*]] = tuple.from_elements %[[V5]], %[[V1]], %[[V9]] : tuple<i32, i32, i32>

  return
  // CHECK-NEXT:     return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
