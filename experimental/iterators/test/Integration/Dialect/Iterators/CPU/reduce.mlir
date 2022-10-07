// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!i32_struct = !llvm.struct<(i32)>

func.func private @sum_struct(%lhs : !i32_struct, %rhs : !i32_struct) -> !i32_struct {
  %lhsi = llvm.extractvalue %lhs[0 : index] : !i32_struct
  %rhsi = llvm.extractvalue %rhs[0 : index] : !i32_struct
  %i = arith.addi %lhsi, %rhsi : i32
  %result = llvm.insertvalue %i, %lhs[0 : index] : !i32_struct
  return %result : !i32_struct
}

func.func @reduce_sum_struct() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<!i32_struct>)
  %reduced = "iterators.reduce"(%input) {reduceFuncRef = @sum_struct}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<!i32_struct>)
  "iterators.sink"(%reduced) : (!iterators.stream<!i32_struct>) -> ()
  // CHECK:      (6)
  return
}

func.func private @sum_i32(%lhs : i32, %rhs : i32) -> i32 {
  %result = arith.addi %lhs, %rhs : i32
  return %result : i32
}

func.func private @unpack_i32(%input : !i32_struct) -> i32 {
  %i = llvm.extractvalue %input[0 : index] : !i32_struct
  return %i : i32
}

func.func private @pack_i32(%input : i32) -> !i32_struct {
  %undef = llvm.mlir.undef : !i32_struct
  %result =  llvm.insertvalue %input, %undef[0 : index] : !i32_struct
  return %result : !i32_struct
}

func.func @reduce_sum_i32() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [10 : i32], [20 : i32], [30 : i32]] }
      : () -> (!iterators.stream<!i32_struct>)
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<i32>)
  %reduced = "iterators.reduce"(%unpacked) {reduceFuncRef = @sum_i32}
    : (!iterators.stream<i32>) -> (!iterators.stream<i32>)
  %repacked = "iterators.map"(%reduced) {mapFuncRef = @pack_i32}
    : (!iterators.stream<i32>) -> (!iterators.stream<!i32_struct>)
  "iterators.sink"(%repacked) : (!iterators.stream<!i32_struct>) -> ()
  // CHECK:      (60)
  return
}

!i32f32_struct = !llvm.struct<(i32, f32)>

// Return input where second struct field is larger. Return lhs on equality or
// unordered.
func.func private @arg_max(%lhs : !i32f32_struct, %rhs : !i32f32_struct) -> !i32f32_struct {
  %lhsf = llvm.extractvalue %lhs[1 : index] : !i32f32_struct
  %rhsf = llvm.extractvalue %rhs[1 : index] : !i32f32_struct
  %cmp = arith.cmpf "uge", %lhsf, %rhsf : f32
  %result = scf.if %cmp -> !i32f32_struct {
    scf.yield %lhs : !i32f32_struct
  } else {
    scf.yield %rhs : !i32f32_struct
  }
  return %result : !i32f32_struct
}

func.func @reduce_arg_max() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32,  0.   : f32],
                 [1 : i32, 13.37 : f32],  // <-- max value
                 [2 : i32,  4.2  : f32]] }
      : () -> (!iterators.stream<!i32f32_struct>)
  %reduce = "iterators.reduce"(%input) {reduceFuncRef = @arg_max}
    : (!iterators.stream<!i32f32_struct>) -> (!iterators.stream<!i32f32_struct>)
  "iterators.sink"(%reduce) : (!iterators.stream<!i32f32_struct>) -> ()
  // CHECK:      (1, 13.37)
  return
}

func.func @main() {
  call @reduce_sum_struct() : () -> ()
  call @reduce_sum_i32() : () -> ()
  call @reduce_arg_max() : () -> ()
  return
}
