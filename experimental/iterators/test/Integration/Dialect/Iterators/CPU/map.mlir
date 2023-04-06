// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -decompose-tuples \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func private @double_tuple(%tuple : tuple<i32>) -> tuple<i32> {
  %i = tuple.to_elements %tuple : tuple<i32>
  %doubled = arith.addi %i, %i : i32
  %result = tuple.from_elements %doubled : tuple<i32>
  return %result : tuple<i32>
}

func.func @map_double_tuple() {
  iterators.print("map_double_tuple")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %reduce = "iterators.map"(%input) {mapFuncRef = @double_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  "iterators.sink"(%reduce) : (!iterators.stream<tuple<i32>>) -> ()
  // CHECK-LABEL: map_double_tuple
  // CHECK-NEXT:  (0)
  // CHECK-NEXT:  (2)
  // CHECK-NEXT:  (4)
  // CHECK-NEXT:  (6)
  // CHECK-NEXT:  -
  return
}

func.func private @add_field(%input : tuple<i32>) -> tuple<i32, f32> {
  %i = tuple.to_elements %input : tuple<i32>
  %f = arith.sitofp %i : i32 to f32
  %result = tuple.from_elements %i, %f : tuple<i32, f32>
  return %result : tuple<i32, f32>
}

func.func @map_add_field() {
  iterators.print("map_add_field")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %mapped = "iterators.map"(%input) {mapFuncRef = @add_field}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32, f32>>)
  "iterators.sink"(%mapped) : (!iterators.stream<tuple<i32, f32>>) -> ()
  // CHECK-LABEL: map_add_field
  // CHECK-NEXT:  (0, 0)
  // CHECK-NEXT:  (1, 1)
  // CHECK-NEXT:  (2, 2)
  // CHECK-NEXT:  -
  return
}

func.func private @unpack_i32(%input : tuple<i32>) -> i32 {
  %i = tuple.to_elements %input : tuple<i32>
  return %i : i32
}

func.func @map_unpack_pack() {
  iterators.print("map_unpack_pack")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)
  "iterators.sink"(%unpacked) : (!iterators.stream<i32>) -> ()
  // CHECK-LABEL: map_unpack_pack
  // CHECK-NEXT:  0
  // CHECK-NEXT:  1
  // CHECK-NEXT:  2
  // CHECK-NEXT:  -
  return
}

func.func @main() {
  call @map_double_tuple() : () -> ()
  call @map_add_field() : () -> ()
  call @map_unpack_pack() : () -> ()
  return
}
