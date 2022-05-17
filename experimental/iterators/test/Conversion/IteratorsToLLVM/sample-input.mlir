// RUN: mlir-proto-opt %s -convert-iterators-to-llvm \
// RUN: | FileCheck %s

!intTuple = type !llvm.struct<(i32)>

// CHECK: 	   module {
// CHECK-NEXT:   func private @[[closeFunctionName:.*]](%[[arg0:.*]]: !llvm.struct<"[[inputStateType:.*]]", (i32)>) -> !llvm.struct<"[[inputStateType]]", (i32)> {
// CHECK-NEXT:     return %[[arg0]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @[[nextFunctionName:.*]](%[[arg1:.*]]: !llvm.struct<"[[inputStateType]]", (i32)>) -> (!llvm.struct<"[[inputStateType]]", (i32)>, i1, !llvm.struct<(i32)>) {
// CHECK-NEXT:     %[[V0:.*]] = llvm.extractvalue %[[arg1]][0 : index] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:     %[[V1:.*]] = arith.constant 4 : i32
// CHECK-NEXT:     %[[V2:.*]] = arith.cmpi slt, %[[V0]], %[[V1]] : i32
// CHECK-NEXT:     %[[V3:.*]] = scf.if %[[V2]] -> (!llvm.struct<"[[inputStateType]]", (i32)>) {
// CHECK-NEXT:       %[[V4:.*]] = arith.constant 1 : i32
// CHECK-NEXT:       %[[V5:.*]] = arith.addi %[[V0]], %[[V4]] : i32
// CHECK-NEXT:       %[[V6:.*]] = llvm.insertvalue %[[V5]], %[[arg1]][0 : index] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:       scf.yield %[[V6]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:     } else {
// CHECK-NEXT:       scf.yield %[[arg1]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:     }
// CHECK-NEXT:     %[[V7:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
// CHECK-NEXT:     %[[V8:.*]] = llvm.insertvalue %[[V0]], %[[V7]][0 : index] : !llvm.struct<(i32)>
// CHECK-NEXT:     return %[[V3]], %[[V2]], %[[V8]] : !llvm.struct<"[[inputStateType]]", (i32)>, i1, !llvm.struct<(i32)>
// CHECK-NEXT:   }
// CHECK-NEXT:   func private @[[openFunctionName:.*]](%[[arg2:.*]]: !llvm.struct<"[[inputStateType]]", (i32)>) -> !llvm.struct<"[[inputStateType]]", (i32)> {
// CHECK-NEXT:     %[[V9:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK-NEXT:     %[[Va:.*]] = llvm.insertvalue %[[V9]], %[[arg2]][0 : index] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:     return %[[Va]] : !llvm.struct<"[[inputStateType]]", (i32)>
// CHECK-NEXT:   }
func @main() {
  // CHECK-NEXT: func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<!intTuple>)
  // CHECK-NEXT:   %[[Vb:.*]] = llvm.mlir.undef : !llvm.struct<"[[inputStateType]]", (i32)>
  // CHECK-NEXT:   return
  return
  // CHECK-NEXT: }
}
// CHECK-NEXT: }
