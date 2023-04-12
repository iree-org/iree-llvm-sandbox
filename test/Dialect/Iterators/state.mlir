// RUN: iterators-opt %s \
// RUN: | FileCheck %s

func.func @testUndefInsertExtract() {
// CHECK-LABEL: func.func @testUndefInsertExtract() {
  %value = arith.constant 0 : i32
// CHECK-NEXT:     %[[V1:.*]] = arith.constant 0 : i32
  %initial_state = iterators.createstate(%value) : !iterators.state<i32>
// CHECK-NEXT:     %[[V0:state.*]] = iterators.createstate(%[[V1]]) : !iterators.state<i32>
  %inserted_state = iterators.insertvalue %value into
                        %initial_state[0] : !iterators.state<i32>
// CHECK-NEXT:     %[[V2:state.*]] = iterators.insertvalue %[[V1]] into %[[V0]][0] : !iterators.state<i32>
  %extracted_value = iterators.extractvalue %inserted_state[0] :
                         !iterators.state<i32>
// CHECK-NEXT:     %[[V3:.*]] = iterators.extractvalue %[[V2]][0] : !iterators.state<i32>
  return
// CHECK-NEXT:     return
}
// CHECK-NEXT:   }
