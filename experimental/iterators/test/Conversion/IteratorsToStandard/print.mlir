// RUN: mlir-proto-opt %s -convert-iterators-to-std \
// RUN: | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   llvm.func @printf(!llvm.ptr<i8>, ...) -> i32
// CHECK-NEXT:   llvm.mlir.global internal constant @frmt_spec.tuple[[S0:.*]]("(%i)\0A\00")
func @main() {
  // CHECK-NEXT:   func @main() {

  %oneFieldTuple = "iterators.constant"() { values = [1 : i32] } : () -> tuple<i32>
  // CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S0]]", (i32)>
  // CHECK-NEXT:     %[[V1:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:     %[[V2:.*]] = llvm.insertvalue %[[V1]], %[[V0]][0 : index] : !llvm.struct<"tuple[[S0]]", (i32)>

  "iterators.print"(%oneFieldTuple) : (tuple<i32>) -> ()
  // CHECK:      %[[V3:.*]] = llvm.mlir.addressof @frmt_spec.tuple[[S0]] : !llvm.ptr<array<6 x i8>>
  // CHECK-NEXT: %[[V4:.*]] = llvm.mlir.constant(0 : i64) : i64
  // CHECK-NEXT: %[[V5:.*]] = llvm.getelementptr %[[V3]][%[[V4]], %[[V4]]] : (!llvm.ptr<array<6 x i8>>, i64, i64) -> !llvm.ptr<i8>
  // CHECK-NEXT: %[[V6:.*]] = llvm.extractvalue %[[V2]][0 : index] : !llvm.struct<"tuple[[S0]]", (i32)>
  // CHECK-NEXT: %[[V7:.*]] = llvm.call @printf(%[[V5]], %[[V6]]) : (!llvm.ptr<i8>, i32) -> i32
  return
  // CHECK-NEXT:     return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
