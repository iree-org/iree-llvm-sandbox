// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
// CHECK-NEXT:   llvm.mlir.global internal constant @frmt_spec.tuple[[S0:.*]]("(%lli)\0A\00")
// CHECK-NEXT:   func.func @main() {
func.func @main() {
  %empty_tuple = "iterators.constanttuple"() { values = [] } : () -> tuple<>
  // CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S0:.*]]", ()>

  %one_field_tuple = "iterators.constanttuple"() { values = [1 : i32] } : () -> tuple<i32>
  // CHECK-NEXT:     %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S1:.*]]", (i32)>
  // CHECK-NEXT:     %[[V2:.*]] = arith.constant 1 : i32
  // CHECK-NEXT:     %[[V3:.*]] = llvm.insertvalue %[[V2]], %[[V1]][0 : index] : !llvm.struct<"tuple[[S1]]", (i32)>

  "iterators.printtuple"(%one_field_tuple) : (tuple<i32>) -> ()
  // CHECK:          %[[V4:.*]] = llvm.mlir.addressof @frmt_spec.tuple[[S1]] : !llvm.ptr<array<8 x i8>>
  // CHECK-NEXT:     %[[V5:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT:     %[[V6:.*]] = llvm.getelementptr %[[V4]][%[[V5]], %[[V5]]] : (!llvm.ptr<array<8 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK-NEXT:     %[[V7:.*]] = llvm.extractvalue %[[V3]][0 : index] : !llvm.struct<"tuple[[S1]]", (i32)>
  // CHECK-NEXT:     %[[Vb:.*]] = llvm.zext %[[V7]] : i32 to i64
  // CHECK-NEXT:     %[[V8:.*]] = llvm.call @printf(%[[V6]], %[[Vb]]) : (!llvm.ptr<i8>, i64) -> i32

  %three_field_tuple = "iterators.constanttuple"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  // CHECK-NEXT:     %[[V4:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S2:.*]]", (i32, i32, i32)>
  // CHECK-NEXT:     %[[V5:.*]] = arith.constant 1 : i32
  // CHECK-NEXT:     %[[V6:.*]] = llvm.insertvalue %[[V5]], %[[V4]][0 : index] : !llvm.struct<"tuple[[S2]]", (i32, i32, i32)>
  // CHECK-NEXT:     %[[V7:.*]] = arith.constant 2 : i32
  // CHECK-NEXT:     %[[V8:.*]] = llvm.insertvalue %[[V7]], %[[V6]][1 : index] : !llvm.struct<"tuple[[S2]]", (i32, i32, i32)>
  // CHECK-NEXT:     %[[V9:.*]] = arith.constant 3 : i32
  // CHECK-NEXT:     %[[Va:.*]] = llvm.insertvalue %[[V9]], %[[V8]][2 : index] : !llvm.struct<"tuple[[S2]]", (i32, i32, i32)>

  return
  // CHECK-NEXT:     return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
