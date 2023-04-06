// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -decompose-tuples \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func private @sum_tuple(%lhs : tuple<i32>, %rhs : tuple<i32>) -> tuple<i32> {
  %lhsi = tuple.to_elements %lhs : tuple<i32>
  %rhsi = tuple.to_elements %rhs : tuple<i32>
  %i = arith.addi %lhsi, %rhsi : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

func.func @reduce_sum_tuple() {
  iterators.print("reduce_sum_tuple")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %reduced = "iterators.reduce"(%input) {reduceFuncRef = @sum_tuple}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  "iterators.sink"(%reduced) : (!iterators.stream<tuple<i32>>) -> ()
  // CHECK-LABEL: reduce_sum_tuple
  // CHECK-NEXT:  (6)
  // CHECK-NEXT:  -
  return
}

func.func private @sum_i32(%lhs : i32, %rhs : i32) -> i32 {
  %result = arith.addi %lhs, %rhs : i32
  return %result : i32
}

func.func private @unpack_i32(%input : tuple<i32>) -> i32 {
  %i = tuple.to_elements %input : tuple<i32>
  return %i : i32
}

func.func @reduce_sum_i32() {
  iterators.print("reduce_sum_i32")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [10 : i32], [20 : i32], [30 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)
  %reduced = "iterators.reduce"(%unpacked) {reduceFuncRef = @sum_i32}
    : (!iterators.stream<i32>) -> (!iterators.stream<i32>)
  "iterators.sink"(%reduced) : (!iterators.stream<i32>) -> ()
  // CHECK-LABEL: reduce_sum_i32
  // CHECK-NEXT:  60
  // CHECK-NEXT:  -
  return
}

// Return input where second tuple element is larger. Return lhs on equality or
// unordered.
func.func private @arg_max(%lhs : tuple<i32, f32>, %rhs : tuple<i32, f32>) -> tuple<i32, f32> {
  %lhsi, %lhsf = tuple.to_elements %lhs : tuple<i32, f32>
  %rhsi, %rhsf = tuple.to_elements %rhs : tuple<i32, f32>
  %cmp = arith.cmpf "uge", %lhsf, %rhsf : f32
  %result = scf.if %cmp -> tuple<i32, f32> {
    scf.yield %lhs : tuple<i32, f32>
  } else {
    scf.yield %rhs : tuple<i32, f32>
  }
  return %result : tuple<i32, f32>
}

func.func @reduce_arg_max() {
  iterators.print("reduce_arg_max")
  %input = "iterators.constantstream"()
      { value = [[0 : i32,  0.   : f32],
                 [1 : i32, 13.37 : f32],  // <-- max value
                 [2 : i32,  4.2  : f32]] }
      : () -> (!iterators.stream<tuple<i32, f32>>)
  %reduce = "iterators.reduce"(%input) {reduceFuncRef = @arg_max}
    : (!iterators.stream<tuple<i32, f32>>) -> (!iterators.stream<tuple<i32, f32>>)
  "iterators.sink"(%reduce) : (!iterators.stream<tuple<i32, f32>>) -> ()
  // CHECK-LABEL: reduce_arg_max
  // CHECK-NEXT:  (1, 13.37)
  // CHECK-NEXT:  -
  return
}

func.func @main() {
  call @reduce_sum_tuple() : () -> ()
  call @reduce_sum_i32() : () -> ()
  call @reduce_arg_max() : () -> ()
  return
}
