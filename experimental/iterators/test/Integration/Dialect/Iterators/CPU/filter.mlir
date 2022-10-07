// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!i32_struct = !llvm.struct<(i32)>

func.func private @is_positive_struct(%struct : !i32_struct) -> i1 {
  %i = llvm.extractvalue %struct[0 : index] : !i32_struct
  %zero = arith.constant 0 : i32
  %cmp = arith.cmpi "sgt", %i, %zero : i32
  return %cmp : i1
}

func.func @filter_is_positive_struct() {
  %input = "iterators.constantstream"()
    { value = [[0: i32], [1: i32], [-1: i32], [2: i32], [-2: i32]] }
    : () -> (!iterators.stream<!i32_struct>)
  %filtered = "iterators.filter"(%input) {predicateRef = @is_positive_struct}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<!i32_struct>)
  "iterators.sink"(%filtered) : (!iterators.stream<!i32_struct>) -> ()
  // CHECK:      (1)
  // CHECK-NEXT: (2)
  return
}

func.func private @is_positive_i32(%i : i32) -> i1 {
  %zero = arith.constant 0 : i32
  %cmp = arith.cmpi "sgt", %i, %zero : i32
  return %cmp : i1
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

func.func @filter_is_positive_i32() {
  %input = "iterators.constantstream"()
    { value = [[0: i32], [2: i32], [-2: i32], [4: i32], [-4: i32]] }
    : () -> (!iterators.stream<!i32_struct>)
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<i32>)
  %filtered = "iterators.filter"(%unpacked) {predicateRef = @is_positive_i32}
    : (!iterators.stream<i32>) -> (!iterators.stream<i32>)
  %repacked = "iterators.map"(%filtered) {mapFuncRef = @pack_i32}
    : (!iterators.stream<i32>) -> (!iterators.stream<!i32_struct>)
  "iterators.sink"(%repacked) : (!iterators.stream<!i32_struct>) -> ()
  // CHECK:      (2)
  // CHECK-NEXT: (4)
  return
}

func.func @main() {
  call @filter_is_positive_struct() : () -> ()
  call @filter_is_positive_i32() : () -> ()
  return
}
