// Test that we can parse and verify ops on iterator state correctly, and that
// they round-trip through assembly.
// RUN: mlir-proto-opt %s \
// RUN: | FileCheck %s

func.func @testUndefInsertExtract() {
// CHECK-LABEL: func.func @testUndefInsertExtract() {
  %initial_state = iterators.undefstate : !iterators.state<i32>
// CHECK-NEXT:     %[[V0:.*]] = iterators.undefstate : <i32>
  %value = arith.constant 0 : i32
// CHECK-NEXT:     %[[V1:.*]] = arith.constant 0 : i32
  %inserted_state = iterators.insertvalue %initial_state[0] (%value : i32) : !iterators.state<i32>
// CHECK-NEXT:     %[[V2:.*]] = iterators.insertvalue %[[V0]][0] (%[[V1]] : i32) : <i32>
  %extracted_value = iterators.extractvalue %inserted_state[0] : !iterators.state<i32> -> i32
// CHECK-NEXT:     %[[V3:.*]] = iterators.extractvalue %[[V2]][0] : <i32> -> i32
  return
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }
