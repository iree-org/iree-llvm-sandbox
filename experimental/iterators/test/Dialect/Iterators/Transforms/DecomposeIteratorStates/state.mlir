// RUN: iterators-opt %s -decompose-iterator-states \
// RUN: | FileCheck --enable-var-scope %s

// CHECK-LABEL: func.func @testCreateInsertExtractFlat() -> (i32, i32) {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    return %[[V0]], %[[V0]] : i32, i32
// CHECK-NEXT:  }
func.func @testCreateInsertExtractFlat() -> (i32, i32) {
  %value = arith.constant 0 : i32
  %created_state = iterators.createstate(%value) : !iterators.state<i32>
  %extracted_value_0 = iterators.extractvalue %created_state[0] : !iterators.state<i32>
  %inserted_state = iterators.insertvalue %value into %created_state[0] : !iterators.state<i32>
  %extracted_value_1 = iterators.extractvalue %inserted_state[0] : !iterators.state<i32>
  return %extracted_value_0, %extracted_value_1 : i32, i32
}

// CHECK-LABEL: func.func @testCreateInsertExtractNested() -> (i32, i32, i32, i32) {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    return %[[V0]], %[[V0]], %[[V0]], %[[V0]] : i32, i32, i32, i32
// CHECK-NEXT:  }
func.func @testCreateInsertExtractNested() -> (i32, i32, i32, i32) {
  // Create nested states.
  %value = arith.constant 0 : i32
  %created_inner_state = iterators.createstate(%value) : !iterators.state<i32>
  %created_outer_state = iterators.createstate(%value, %created_inner_state) : !iterators.state<i32, !iterators.state<i32>>
  // Extract from created state.
  %extracted_value_0 = iterators.extractvalue %created_outer_state[0] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_inner_state_0 = iterators.extractvalue %created_outer_state[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_1 = iterators.extractvalue %extracted_inner_state_0[0] : !iterators.state<i32>
  // Update states and then extract again.
  %inserted_inner_state = iterators.insertvalue %value into %created_inner_state[0] : !iterators.state<i32>
  %inserted_outer_state_0 = iterators.insertvalue %value into %created_outer_state[0] : !iterators.state<i32, !iterators.state<i32>>
  %inserted_outer_state_1 = iterators.insertvalue %inserted_inner_state into %inserted_outer_state_0[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_2 = iterators.extractvalue %inserted_outer_state_1[0] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_inner_state_1 = iterators.extractvalue %inserted_outer_state_1[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_3 = iterators.extractvalue %extracted_inner_state_1[0] : !iterators.state<i32>
  return %extracted_value_0, %extracted_value_1, %extracted_value_2, %extracted_value_3 : i32, i32, i32, i32
}

// CHECK-LABEL: func.func @testSCFIfNested(
// CHECK-SAME:        %[[arg0:.*]]: i1) -> (i32, i32) {
// CHECK-NEXT:    %[[V1:.*]] = arith.constant 1 : i32
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[V3:.*]]:2 = scf.if %[[arg0]] -> (i32, i32) {
// CHECK-NEXT:      scf.yield %[[V1]], %[[V1]] : i32, i32
// CHECK-NEXT:    } else {
// CHECK-NEXT:      scf.yield %[[V0]], %[[V0]] : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V3]]#0, %[[V3]]#1 : i32, i32
// CHECK-NEXT:  }
func.func @testSCFIfNested(%do_insert : i1) -> (i32, i32) {
  // Create nested state.
  %init_value = arith.constant 0 : i32
  %created_inner_state = iterators.createstate(%init_value) : !iterators.state<i32>
  %created_outer_state = iterators.createstate(%init_value, %created_inner_state) : !iterators.state<i32, !iterators.state<i32>>
  // Possibly insert new values.
  %updated_outer_state = scf.if %do_insert -> !iterators.state<i32, !iterators.state<i32>> {
    %insert_value = arith.constant 1 : i32
    %inserted_inner_state = iterators.insertvalue %insert_value into %created_inner_state[0] : !iterators.state<i32>
    %inserted_outer_state = iterators.createstate(%insert_value, %inserted_inner_state) : !iterators.state<i32, !iterators.state<i32>>
    scf.yield %inserted_outer_state : !iterators.state<i32, !iterators.state<i32>>
  } else {
    scf.yield %created_outer_state : !iterators.state<i32, !iterators.state<i32>>
  }
  // Extract possibly overwritten values.
  %extracted_value_0 = iterators.extractvalue %updated_outer_state[0] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_inner_state = iterators.extractvalue %updated_outer_state[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_1 = iterators.extractvalue %extracted_inner_state[0] : !iterators.state<i32>
  return %extracted_value_0, %extracted_value_1 : i32, i32
}

// CHECK-LABEL: func.func @testSCFWhileNested(
// CHECK-SAME:        %[[arg0:.*]]: i32) -> (i32, i32) {
// CHECK-NEXT:    %[[V0:.*]] = arith.constant 0 : i32
// CHECK-NEXT:    %[[V1:.*]]:2 = scf.while (%[[arg1:.*]] = %[[V0]], %[[arg2:.*]] = %[[V0]]) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:      %[[V2:.*]] = arith.cmpi eq, %[[arg0]], %[[arg1]] : i32
// CHECK-NEXT:      scf.condition(%[[V2]]) %[[arg1]], %[[arg2]] : i32, i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%[[arg3:.*]]: i32, %[[arg4:.*]]: i32):
// CHECK-NEXT:      scf.yield %[[arg3]], %[[V0]] : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V1]]#0, %[[V1]]#1 : i32, i32
// CHECK-NEXT:  }
func.func @testSCFWhileNested(%arg : i32) -> (i32, i32) {
  // Create nested state.
  %init_value = arith.constant 0 : i32
  %created_inner_state = iterators.createstate(%init_value) : !iterators.state<i32>
  %created_outer_state = iterators.createstate(%init_value, %created_inner_state) : !iterators.state<i32, !iterators.state<i32>>
  // Do dummy update of state in a loop.
  %processed_outer_state = scf.while (%state = %created_outer_state) : (!iterators.state<i32, !iterators.state<i32>>) -> !iterators.state<i32, !iterators.state<i32>> {
    %extracted_value = iterators.extractvalue %state[0] : !iterators.state<i32, !iterators.state<i32>>
    %cmp = arith.cmpi "eq", %arg, %extracted_value : i32
    scf.condition (%cmp) %state : !iterators.state<i32, !iterators.state<i32>>
  } do {
  ^bb0(%state : !iterators.state<i32, !iterators.state<i32>>):
    %updated_state = iterators.insertvalue %created_inner_state into %state[1] : !iterators.state<i32, !iterators.state<i32>>
    scf.yield %updated_state : !iterators.state<i32, !iterators.state<i32>>
  }
  // Extract nested values.
  %extracted_value_0 = iterators.extractvalue %processed_outer_state[0] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_inner_state = iterators.extractvalue %processed_outer_state[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_1 = iterators.extractvalue %extracted_inner_state[0] : !iterators.state<i32>
  return %extracted_value_0, %extracted_value_1 : i32, i32
}

// CHECK-LABEL: func.func private @testFuncFunc(
// CHECK-SAME:        i32, i32) -> (i32, i32)
func.func private @testFuncFunc(%arg : !iterators.state<i32, !iterators.state<i32>>) -> (!iterators.state<i32, !iterators.state<i32>>)

// CHECK-LABEL: func.func @testFuncBlock(
// CHECK-SAME:        %[[arg0:.*]]: i32, %[[arg1:.*]]: i32) -> (i32, i32) {
// CHECK-NEXT:    return %[[arg0]], %[[arg1]] : i32, i32
// CHECK-NEXT:  }
func.func @testFuncBlock(%arg : !iterators.state<i32, !iterators.state<i32>>) -> (i32, i32) {
  %extracted_value_0 = iterators.extractvalue %arg[0] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_inner_state = iterators.extractvalue %arg[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_1 = iterators.extractvalue %extracted_inner_state[0] : !iterators.state<i32>
  return %extracted_value_0, %extracted_value_1 : i32, i32
}

// CHECK-LABEL: func.func @testFuncCall(
// CHECK-SAME:        %[[arg0:.*]]: i32) -> (i32, i32) {
// CHECK-NEXT:    %[[V0:.*]]:2 = call @testFuncFunc(%[[arg0]], %[[arg0]]) : (i32, i32) -> (i32, i32)
// CHECK-NEXT:    return %[[V0]]#0, %[[V0]]#1 : i32, i32
// CHECK-NEXT:  }
func.func @testFuncCall(%arg : i32) -> (i32, i32) {
  // Create nested state.
  %created_inner_state = iterators.createstate(%arg) : !iterators.state<i32>
  %created_outer_state = iterators.createstate(%arg, %created_inner_state) : !iterators.state<i32, !iterators.state<i32>>
  // Pass it to function call.
  %call_result = func.call @testFuncFunc(%created_outer_state) : (!iterators.state<i32, !iterators.state<i32>>) -> !iterators.state<i32, !iterators.state<i32>>
  // Extract values from call result.
  %extracted_value_0 = iterators.extractvalue %call_result[0] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_inner_state = iterators.extractvalue %call_result[1] : !iterators.state<i32, !iterators.state<i32>>
  %extracted_value_1 = iterators.extractvalue %extracted_inner_state[0] : !iterators.state<i32>
  return %extracted_value_0, %extracted_value_1 : i32, i32
}

// CHECK-LABEL: func.func @testFuncReturn(
// CHECK-SAME:        %arg0: i32) -> (i32, i32) {
// CHECK-NEXT:    return %arg0, %arg0 : i32, i32
// CHECK-NEXT:  }
func.func @testFuncReturn(%arg : i32) -> !iterators.state<i32, !iterators.state<i32>> {
  // Create nested state.
  %created_inner_state = iterators.createstate(%arg) : !iterators.state<i32>
  %created_outer_state = iterators.createstate(%arg, %created_inner_state) : !iterators.state<i32, !iterators.state<i32>>
  // Return from function.
  return %created_outer_state : !iterators.state<i32, !iterators.state<i32>>
}
