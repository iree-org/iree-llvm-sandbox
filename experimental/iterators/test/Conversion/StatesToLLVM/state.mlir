// RUN: mlir-proto-opt %s -convert-states-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

func.func @testUndefInsertExtract() {
// CHECK-LABEL: func.func @testUndefInsertExtract() {
  %initial_state = iterators.undefstate : !iterators.state<i32>
// CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  %value = arith.constant 0 : i32
// CHECK-NEXT:     %[[V1:.*]] = arith.constant 0 : i32
  %inserted_state = iterators.insertvalue %value into %initial_state[0] : !iterators.state<i32>
// CHECK-NEXT:     %[[V2:.*]] = llvm.insertvalue %[[V1]], %[[V0]][0 : index] : !llvm.struct<(i32)>
  %extracted_value = iterators.extractvalue %inserted_state[0] : !iterators.state<i32>
// CHECK-NEXT:     %[[V3:.*]] = llvm.extractvalue %[[V2]][0 : index] : !llvm.struct<(i32)>
  return
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }

func.func @testNestedType() {
// CHECK-LABEL: func.func @testNestedType() {
  %outer_state = iterators.undefstate : !iterators.state<i32, !iterators.state<i32>>
// CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<(i32, struct<(i32)>)>
  %inner_state = iterators.extractvalue %outer_state[1] : !iterators.state<i32, !iterators.state<i32>>
// CHECK-NEXT:     %[[V1:.*]] = llvm.extractvalue %[[V0]][1 : index] : !llvm.struct<(i32, struct<(i32)>)>
  return
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }

func.func @testFuncReturn(%state: !iterators.state<i32>) -> !iterators.state<i32> {
// CHECK-LABEL: func.func @testFuncReturn(%{{.*}}: !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
  return %state : !iterators.state<i32>
// CHECK-NEXT:     return %[[V0:.*]] : !llvm.struct<(i32)>
}
// CHECK-NEXT:   }

func.func @testCall() {
// CHECK-LABEL: func.func @testCall() {
  %state = iterators.undefstate : !iterators.state<i32>
// CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  func.call @testFuncReturn(%state) : (!iterators.state<i32>) -> !iterators.state<i32>
// CHECK-NEXT:     %[[V1:.*]] = call @testFuncReturn(%[[V0]]) : (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>
  return
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }

func.func @testScf() {
// CHECK-LABEL: func.func @testScf() {
  %state = iterators.undefstate : !iterators.state<i32>
// CHECK-NEXT:     %[[V0:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  %cmp = arith.constant true
// CHECK-NEXT:     %[[V1:.*]] = arith.constant true
  %a = scf.if %cmp -> !iterators.state<i32> {
// CHECK-NEXT:     %[[V2:.*]] = scf.if %[[V1]] -> (!llvm.struct<(i32)>) {
    scf.yield %state :  !iterators.state<i32>
// CHECK-NEXT:       scf.yield %[[V0]] : !llvm.struct<(i32)>
  } else {
// CHECK-NEXT:     } else {
    scf.yield %state :  !iterators.state<i32>
// CHECK-NEXT:       scf.yield %[[V0]] : !llvm.struct<(i32)>
  }
// CHECK-NEXT:     }
  return
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }
