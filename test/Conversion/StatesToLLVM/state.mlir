// RUN: structured-opt %s -convert-states-to-llvm \
// RUN: | FileCheck --enable-var-scope %s

func.func @testUndefInsertExtract() {
// CHECK-LABEL:  func.func @testUndefInsertExtract() {
  %value = arith.constant 0 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 0 : i32
  %initial_state = iterators.createstate(%value) : !iterators.state<i32>
  // CHECK-NEXT:   %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-NEXT:   %[[V2:.*]] = llvm.insertvalue %[[V0]], %[[V1]][0] : !llvm.struct<(i32)>
  %inserted_state = iterators.insertvalue %value into %initial_state[0] : !iterators.state<i32>
  // CHECK-NEXT:   %[[V3:.*]] = llvm.insertvalue %[[V0]], %[[V2]][0] : !llvm.struct<(i32)>
  %extracted_value = iterators.extractvalue %inserted_state[0] : !iterators.state<i32>
  // CHECK-NEXT:   %[[V4:.*]] = llvm.extractvalue %[[V3]][0] : !llvm.struct<(i32)>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }

func.func @testNestedType() {
// CHECK-LABEL:  func.func @testNestedType() {
  %value = arith.constant 0 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 0 : i32
  %inner_state = iterators.createstate(%value) : !iterators.state<i32>
  // CHECK-NEXT:   %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-NEXT:   %[[V2:.*]] = llvm.insertvalue %[[V0]], %[[V1]][0] : !llvm.struct<(i32)>
  %outer_state = iterators.createstate(%value, %inner_state) : !iterators.state<i32, !iterators.state<i32>>
  // CHECK-NEXT:   %[[V3:.*]] = llvm.mlir.undef : !llvm.struct<(i32, struct<(i32)>)>
  // CHECK-NEXT:   %[[V4:.*]] = llvm.insertvalue %[[V0]], %[[V3]][0] : !llvm.struct<(i32, struct<(i32)>)>
  // CHECK-NEXT:   %[[V5:.*]] = llvm.insertvalue %[[V2]], %[[V4]][1] : !llvm.struct<(i32, struct<(i32)>)>
  %extracted_inner_state = iterators.extractvalue %outer_state[1] : !iterators.state<i32, !iterators.state<i32>>
  // CHECK-NEXT:   %[[V6:.*]] = llvm.extractvalue %[[V5]][1] : !llvm.struct<(i32, struct<(i32)>)>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }

func.func @testFuncReturn(%state: !iterators.state<i32>) -> !iterators.state<i32> {
// CHECK-LABEL:  func.func @testFuncReturn(%{{.*}}: !llvm.struct<(i32)>) -> !llvm.struct<(i32)> {
  return %state : !iterators.state<i32>
  // CHECK-NEXT:   return %[[V0:.*]] : !llvm.struct<(i32)>
}
// CHECK-NEXT:   }

func.func @testCall() {
// CHECK-LABEL:  func.func @testCall() {
  %value = arith.constant 0 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 0 : i32
  %state = iterators.createstate(%value) : !iterators.state<i32>
  // CHECK-NEXT:   %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-NEXT:   %[[V2:.*]] = llvm.insertvalue %[[V0]], %[[V1]][0] : !llvm.struct<(i32)>
  func.call @testFuncReturn(%state) : (!iterators.state<i32>) -> !iterators.state<i32>
  // CHECK-NEXT:   %[[V3:.*]] = call @testFuncReturn(%[[V2]]) : (!llvm.struct<(i32)>) -> !llvm.struct<(i32)>
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }

func.func @testScf() {
// CHECK-LABEL: func.func @testScf() {
  %value  = arith.constant 0 : i32
  // CHECK-NEXT:   %[[V0:.*]] = arith.constant 0 : i32
  %state = iterators.createstate(%value) : !iterators.state<i32>
  // CHECK-NEXT:   %[[V1:.*]] = llvm.mlir.undef : !llvm.struct<(i32)>
  // CHECK-NEXT:   %[[V2:.*]] = llvm.insertvalue %[[V0]], %[[V1]][0] : !llvm.struct<(i32)>
  %cmp = arith.constant true
  // CHECK-NEXT:   %[[V3:.*]] = arith.constant true
  %a = scf.if %cmp -> !iterators.state<i32> {
  // CHECK-NEXT:   %[[V4:.*]] = scf.if %[[V3]] -> (!llvm.struct<(i32)>) {
  scf.yield %state :  !iterators.state<i32>
  // CHECK-NEXT:     scf.yield %[[V2]] : !llvm.struct<(i32)>
  } else {
  // CHECK-NEXT:   } else {
  scf.yield %state :  !iterators.state<i32>
  // CHECK-NEXT:     scf.yield %[[V2]] : !llvm.struct<(i32)>
  }
  // CHECK-NEXT:   }
  return
  // CHECK-NEXT:   return
}
// CHECK-NEXT:   }
