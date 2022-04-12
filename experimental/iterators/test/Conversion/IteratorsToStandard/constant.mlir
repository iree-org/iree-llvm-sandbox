// RUN: mlir-proto-opt %s -convert-iterators-to-std \
// RUN: | FileCheck %s

// CHECK:      module {
// CHECK-NEXT:   func @main() {
func @main() {
  %emptyTuple = "iterators.constant"() { values = [] } : () -> tuple<>
  // CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S0:.*]]", ()>

  %oneFieldTuple = "iterators.constant"() { values = [1 : i32] } : () -> tuple<i32>
  // CHECK-NEXT:     %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S1:.*]]", (i32)>
  // CHECK-NEXT:     %[[V2:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:     %[[V3:.*]] = llvm.insertvalue %[[V2]], %[[V1]][0 : index] : !llvm.struct<"tuple[[S1]]", (i32)>

  %threeFieldTuple = "iterators.constant"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  // CHECK-NEXT:     %[[V4:.*]] = llvm.mlir.undef : !llvm.struct<"tuple[[S2:.*]]", (i32, i32, i32)>
  // CHECK-NEXT:     %[[V5:.*]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK-NEXT:     %[[V6:.*]] = llvm.insertvalue %[[V5]], %[[V4]][0 : index] : !llvm.struct<"tuple[[S2]]", (i32, i32, i32)>
  // CHECK-NEXT:     %[[V7:.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK-NEXT:     %[[V8:.*]] = llvm.insertvalue %[[V7]], %[[V6]][1 : index] : !llvm.struct<"tuple[[S2]]", (i32, i32, i32)>
  // CHECK-NEXT:     %[[V9:.*]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK-NEXT:     %[[Va:.*]] = llvm.insertvalue %[[V9]], %[[V8]][2 : index] : !llvm.struct<"tuple[[S2]]", (i32, i32, i32)>

  return
  // CHECK-NEXT:     return
}
// CHECK-NEXT:   }
// CHECK-NEXT: }
