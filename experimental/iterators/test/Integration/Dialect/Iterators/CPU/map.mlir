// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!i32_struct = !llvm.struct<(i32)>

func.func private @double_struct(%struct : !i32_struct) -> !i32_struct {
  %i = llvm.extractvalue %struct[0 : index] : !i32_struct
  %doubled = arith.addi %i, %i : i32
  %result = llvm.insertvalue %doubled, %struct[0 : index] : !i32_struct
  return %result : !i32_struct
}

func.func @map_double_struct() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<!i32_struct>)
  %reduce = "iterators.map"(%input) {mapFuncRef = @double_struct}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<!i32_struct>)
  "iterators.sink"(%reduce) : (!iterators.stream<!i32_struct>) -> ()
  // CHECK:      (0)
  // CHECK-NEXT: (2)
  // CHECK-NEXT: (4)
  // CHECK-NEXT: (6)
  return
}

!i32f32_struct = !llvm.struct<(i32, f32)>

func.func private @add_field(%input : !i32_struct) -> !i32f32_struct {
  %i = llvm.extractvalue %input[0 : index] : !i32_struct
  %f = arith.sitofp %i : i32 to f32
  %undef = llvm.mlir.undef : !i32f32_struct
  %extended =  llvm.insertvalue %i, %undef[0 : index] : !i32f32_struct
  %result =  llvm.insertvalue %f, %extended[1 : index] : !i32f32_struct
  return %result : !i32f32_struct
}

func.func @map_add_field() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32]] }
      : () -> (!iterators.stream<!i32_struct>)
  %mapped = "iterators.map"(%input) {mapFuncRef = @add_field}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<!i32f32_struct>)
  "iterators.sink"(%mapped) : (!iterators.stream<!i32f32_struct>) -> ()
  // CHECK:      (0, 0)
  // CHECK-NEXT: (1, 1)
  // CHECK-NEXT: (2, 2)
  return
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

func.func @map_unpack_pack() {
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32]] }
      : () -> (!iterators.stream<!i32_struct>)
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<!i32_struct>) -> (!iterators.stream<i32>)
  %repacked = "iterators.map"(%unpacked) {mapFuncRef = @pack_i32}
    : (!iterators.stream<i32>) -> (!iterators.stream<!i32_struct>)
  "iterators.sink"(%repacked) : (!iterators.stream<!i32_struct>) -> ()
  // CHECK:      (0)
  // CHECK-NEXT: (1)
  // CHECK-NEXT: (2)
  return
}

func.func @main() {
  call @map_double_struct() : () -> ()
  call @map_add_field() : () -> ()
  call @map_unpack_pack() : () -> ()
  return
}
