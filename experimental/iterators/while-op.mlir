  llvm.mlir.global internal constant @iterators.frmt_spec.0("-\0A\00") {addr_space = 0 : i32}
  llvm.func @printf(!llvm.ptr, ...) -> i32
  llvm.mlir.global internal constant @iterators.frmt_spec("(%llu, %llu)\0A\00") {addr_space = 0 : i32}
  func.func private @iterators.constantstream.close.1(%arg0: !iterators.state<i32>) -> !iterators.state<i32> {
    return %arg0 : !iterators.state<i32>
  }
  llvm.mlir.global internal constant @iterators.constant_stream_data.1() {addr_space = 0 : i32} : !llvm.array<4 x struct<(i32)>> {
    %0 = llvm.mlir.undef : !llvm.array<4 x struct<(i32)>>
    %1 = llvm.mlir.undef : !llvm.struct<(i32)>
    %2 = llvm.mlir.constant(2 : i32) : i32
    %3 = llvm.insertvalue %2, %1[0] : !llvm.struct<(i32)>
    %4 = llvm.insertvalue %3, %0[0] : !llvm.array<4 x struct<(i32)>>
    %5 = llvm.mlir.undef : !llvm.struct<(i32)>
    %6 = llvm.mlir.constant(4 : i32) : i32
    %7 = llvm.insertvalue %6, %5[0] : !llvm.struct<(i32)>
    %8 = llvm.insertvalue %7, %4[1] : !llvm.array<4 x struct<(i32)>>
    %9 = llvm.mlir.undef : !llvm.struct<(i32)>
    %10 = llvm.mlir.constant(6 : i32) : i32
    %11 = llvm.insertvalue %10, %9[0] : !llvm.struct<(i32)>
    %12 = llvm.insertvalue %11, %8[2] : !llvm.array<4 x struct<(i32)>>
    %13 = llvm.mlir.undef : !llvm.struct<(i32)>
    %14 = llvm.mlir.constant(8 : i32) : i32
    %15 = llvm.insertvalue %14, %13[0] : !llvm.struct<(i32)>
    %16 = llvm.insertvalue %15, %12[3] : !llvm.array<4 x struct<(i32)>>
    llvm.return %16 : !llvm.array<4 x struct<(i32)>>
  }
  func.func private @iterators.constantstream.next.1(%arg0: !iterators.state<i32>) -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>) {
    %0 = iterators.extractvalue %arg0[0] : !iterators.state<i32>
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.cmpi slt, %0, %c4_i32 : i32
    %2:2 = scf.if %1 -> (!iterators.state<i32>, !llvm.struct<(i32)>) {
      %c1_i32 = arith.constant 1 : i32
      %3 = arith.addi %0, %c1_i32 : i32
      %state = iterators.insertvalue %3 into %arg0[0] : !iterators.state<i32>
      %4 = llvm.mlir.addressof @iterators.constant_stream_data.1 : !llvm.ptr
      %5 = llvm.getelementptr %4[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32)>
      %6 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(i32)>
      scf.yield %state, %6 : !iterators.state<i32>, !llvm.struct<(i32)>
    } else {
      %3 = llvm.mlir.undef : !llvm.struct<(i32)>
      scf.yield %arg0, %3 : !iterators.state<i32>, !llvm.struct<(i32)>
    }
    return %2#0, %1, %2#1 : !iterators.state<i32>, i1, !llvm.struct<(i32)>
  }
  func.func private @iterators.constantstream.open.1(%arg0: !iterators.state<i32>) -> !iterators.state<i32> {
    %c0_i32 = arith.constant 0 : i32
    %state = iterators.insertvalue %c0_i32 into %arg0[0] : !iterators.state<i32>
    return %state : !iterators.state<i32>
  }
  func.func private @iterators.constantstream.close.0(%arg0: !iterators.state<i32>) -> !iterators.state<i32> {
    return %arg0 : !iterators.state<i32>
  }
  llvm.mlir.global internal constant @iterators.constant_stream_data.0() {addr_space = 0 : i32} : !llvm.array<4 x struct<(i32)>> {
    %0 = llvm.mlir.undef : !llvm.array<4 x struct<(i32)>>
    %1 = llvm.mlir.undef : !llvm.struct<(i32)>
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.insertvalue %2, %1[0] : !llvm.struct<(i32)>
    %4 = llvm.insertvalue %3, %0[0] : !llvm.array<4 x struct<(i32)>>
    %5 = llvm.mlir.undef : !llvm.struct<(i32)>
    %6 = llvm.mlir.constant(1 : i32) : i32
    %7 = llvm.insertvalue %6, %5[0] : !llvm.struct<(i32)>
    %8 = llvm.insertvalue %7, %4[1] : !llvm.array<4 x struct<(i32)>>
    %9 = llvm.mlir.undef : !llvm.struct<(i32)>
    %10 = llvm.mlir.constant(2 : i32) : i32
    %11 = llvm.insertvalue %10, %9[0] : !llvm.struct<(i32)>
    %12 = llvm.insertvalue %11, %8[2] : !llvm.array<4 x struct<(i32)>>
    %13 = llvm.mlir.undef : !llvm.struct<(i32)>
    %14 = llvm.mlir.constant(3 : i32) : i32
    %15 = llvm.insertvalue %14, %13[0] : !llvm.struct<(i32)>
    %16 = llvm.insertvalue %15, %12[3] : !llvm.array<4 x struct<(i32)>>
    llvm.return %16 : !llvm.array<4 x struct<(i32)>>
  }
  func.func private @iterators.constantstream.next.0(%arg0: !iterators.state<i32>) -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>) {
    %0 = iterators.extractvalue %arg0[0] : !iterators.state<i32>
    %c4_i32 = arith.constant 4 : i32
    %1 = arith.cmpi slt, %0, %c4_i32 : i32
    %2:2 = scf.if %1 -> (!iterators.state<i32>, !llvm.struct<(i32)>) {
      %c1_i32 = arith.constant 1 : i32
      %3 = arith.addi %0, %c1_i32 : i32
      %state = iterators.insertvalue %3 into %arg0[0] : !iterators.state<i32>
      %4 = llvm.mlir.addressof @iterators.constant_stream_data.0 : !llvm.ptr
      %5 = llvm.getelementptr %4[%0, 0] : (!llvm.ptr, i32) -> !llvm.ptr, !llvm.struct<(i32)>
      %6 = llvm.load %5 : !llvm.ptr -> !llvm.struct<(i32)>
      scf.yield %state, %6 : !iterators.state<i32>, !llvm.struct<(i32)>
    } else {
      %3 = llvm.mlir.undef : !llvm.struct<(i32)>
      scf.yield %arg0, %3 : !iterators.state<i32>, !llvm.struct<(i32)>
    }
    return %2#0, %1, %2#1 : !iterators.state<i32>, i1, !llvm.struct<(i32)>
  }
  func.func private @iterators.constantstream.open.0(%arg0: !iterators.state<i32>) -> !iterators.state<i32> {
    %c0_i32 = arith.constant 0 : i32
    %state = iterators.insertvalue %c0_i32 into %arg0[0] : !iterators.state<i32>
    return %state : !iterators.state<i32>
  }
  !state_type =
    !iterators.state<
      !iterators.state<i32>,  // lhs state
      !iterators.state<i32>,  // rhs state
      !llvm.struct<(i32)>, i1, // lhs value, hasValue
      !llvm.struct<(i32)>, i1  // rhs value, hasValue
    >
  func.func private @iterators.constantstream.close.2(%arg0: !state_type) -> !state_type {
    %lhs_state = iterators.extractvalue %arg0[0] : !state_type
    %rhs_state = iterators.extractvalue %arg0[1] : !state_type
    %0 = call @iterators.constantstream.close.0(%lhs_state) : (!iterators.state<i32>) -> !iterators.state<i32>
    %1 = call @iterators.constantstream.close.0(%rhs_state) : (!iterators.state<i32>) -> !iterators.state<i32>
    %state_0 = iterators.insertvalue %0 into %arg0[0] : !state_type
    %state_1 = iterators.insertvalue %1 into %state_0[0] : !state_type
    return %state_1 : !state_type
  }
  func.func private @iterators.constantstream.next.2(%arg0: !state_type) -> (!state_type, i1, !llvm.struct<(i32, i32)>) {
    // Pseudocode:
    //   value = undef
    //   hasValue = false
    //   if !lhsHasValue:
    //      lhsState, lhsHasValue, lhsValue = next(lhsState)
    //   if !rhsHasValue:
    //      rhsState, rhsHasValue, rhsValue = next(rhsState)
    //   while (lhsHasValue && rhsHasValue)
    //      if lhsValue < rhsValue:
    //        lhsState, lhsHasValue, lhsValue = next(lhsState)
    //        // return half-undef tuple for outer join
    //        continue
    //      if lhsValue > rhsValue:
    //        rhsState, rhsHasValue, rhsValue = next(rhsState)
    //        // return half-undef tuple for outer join
    //        continue
    //      // assert (lhsValue == rhsValue)
    //      value = tuple(lhsValue, rhsValue)
    //      hasValue = true
    //      lhsHasValue = false
    //      rhsHasValue = false
    //      break
    //   return state, hasValue, value

    // Fetch initial upstream elements if required.
    %initialLhsState = iterators.extractvalue %arg0[0] : !state_type
    %initialRhsState = iterators.extractvalue %arg0[1] : !state_type
    %initialLhsHasValue = iterators.extractvalue %arg0[3] : !state_type
    %initialRhsHasValue = iterators.extractvalue %arg0[5] : !state_type
    %updatedLhsState, %lhsHasValue, %lhsValue = scf.if %initialLhsHasValue -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>) {
      %initialLhsValue = iterators.extractvalue %arg0[2] : !state_type
      scf.yield %initialLhsState, %initialLhsHasValue, %initialLhsValue : !iterators.state<i32>, i1, !llvm.struct<(i32)>
    } else {
      %nextResult:3 = func.call @iterators.constantstream.next.0(%initialLhsState) : (!iterators.state<i32>) -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>)
      scf.yield %nextResult#0, %nextResult#1, %nextResult#2 : !iterators.state<i32>, i1, !llvm.struct<(i32)>
    }
    %updatedRhsState, %rhsHasValue, %rhsValue = scf.if %initialRhsHasValue -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>) {
      %initialRhsValue = iterators.extractvalue %arg0[2] : !state_type
      scf.yield %initialRhsState, %initialRhsHasValue, %initialRhsValue : !iterators.state<i32>, i1, !llvm.struct<(i32)>
    } else {
      %nextResult:3 = func.call @iterators.constantstream.next.0(%initialRhsState) : (!iterators.state<i32>) -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>)
      scf.yield %nextResult#0, %nextResult#1, %nextResult#2 : !iterators.state<i32>, i1, !llvm.struct<(i32)>
    }

    // Main while loop looking for a match.
    %finalLhsState, %finalLhsHasValue, %finalLhsValue, %finalRhsState, %finalRhsHasValue, %finalRhsValue =
      scf.while(%loopLhsState = %updatedLhsState, %loopLhsHasValue = %lhsHasValue, %loopLhsValue = %lhsValue,
                %loopRhsState = %updatedRhsState, %loopRhsHasValue = %rhsHasValue, %loopRhsValue = %rhsValue)
          : (!iterators.state<i32>, i1, !llvm.struct<(i32)>, !iterators.state<i32>, i1, !llvm.struct<(i32)>)
          -> (!iterators.state<i32>, i1, !llvm.struct<(i32)>, !iterators.state<i32>, i1, !llvm.struct<(i32)>) {
        scf.condition
      } do {

      }


    %0 = llvm.mlir.undef : !llvm.struct<(i32, i32)>
    %1 = arith.constant false
    return %arg0, %1, %0 : !state_type, i1, !llvm.struct<(i32, i32)>
  }
  func.func private @iterators.constantstream.open.2(%arg0: !state_type) -> !state_type {
    %lhs_state = iterators.extractvalue %arg0[0] : !state_type
    %rhs_state = iterators.extractvalue %arg0[1] : !state_type
    %0 = call @iterators.constantstream.open.0(%lhs_state) : (!iterators.state<i32>) -> !iterators.state<i32>
    %1 = call @iterators.constantstream.open.0(%rhs_state) : (!iterators.state<i32>) -> !iterators.state<i32>
    %false = arith.constant false
    %state_0 = iterators.insertvalue %0 into %arg0[0] : !state_type
    %state_1 = iterators.insertvalue %1 into %state_0[1] : !state_type
    %state_2 = iterators.insertvalue %false into %state_1[3] : !state_type
    %state_3 = iterators.insertvalue %false into %state_2[5] : !state_type
    return %state_3 : !state_type
  }
  func.func @main() {
    %c0_i32 = arith.constant 0 : i32
    %state = iterators.createstate(%c0_i32) : !iterators.state<i32>
    %state_1 = iterators.createstate(%c0_i32) : !iterators.state<i32>
    %undef = llvm.mlir.undef : !llvm.struct<(i32)>
    %false = arith.constant false
    %state_3 = iterators.createstate(%state, %state_1, %undef, %false, %undef, %false) : !state_type
    %0 = call @iterators.constantstream.open.2(%state_3) : (!state_type) -> !state_type
    %1:2 = scf.while (%arg0 = %0) : (!state_type) -> (!state_type, !llvm.struct<(i32, i32)>) {
      %6:3 = func.call @iterators.constantstream.next.2(%arg0) : (!state_type) -> (!state_type, i1, !llvm.struct<(i32, i32)>)
      scf.condition(%6#1) %6#0, %6#2 : !state_type, !llvm.struct<(i32, i32)>
    } do {
    ^bb0(%arg0: !state_type, %arg1: !llvm.struct<(i32, i32)>):
      %6 = llvm.extractvalue %arg1[0] : !llvm.struct<(i32, i32)>
      %7 = arith.extui %6 : i32 to i64
      %8 = llvm.extractvalue %arg1[1] : !llvm.struct<(i32, i32)>
      %9 = arith.extui %8 : i32 to i64
      %10 = llvm.mlir.addressof @iterators.frmt_spec : !llvm.ptr
      %11 = llvm.getelementptr %10[0] : (!llvm.ptr) -> !llvm.ptr, i8
      %12 = llvm.call @printf(%11, %7, %9) : (!llvm.ptr, i64, i64) -> i32
      scf.yield %arg0 : !state_type
    }
    %2 = call @iterators.constantstream.close.2(%1#0) : (!state_type) -> !state_type
    %3 = llvm.mlir.addressof @iterators.frmt_spec.0 : !llvm.ptr
    %4 = llvm.getelementptr %3[0] : (!llvm.ptr) -> !llvm.ptr, i8
    %5 = llvm.call @printf(%4) : (!llvm.ptr) -> i32
    return
  }
