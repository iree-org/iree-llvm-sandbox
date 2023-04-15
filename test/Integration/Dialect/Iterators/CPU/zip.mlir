// RUN: structured-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-tuples \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func private @unpack_i32(%input : tuple<i32>) -> i32 {
  %i = tuple.to_elements %input : tuple<i32>
  return %i : i32
}

func.func @main() {
  // Left-hand stream of numbers.
  %zero_to_three = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %unpacked_lhs = "iterators.map"(%zero_to_three) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)

  // Right-hand stream of numbers.
  %four_to_seven = "iterators.constantstream"()
      { value = [[4 : i32], [5 : i32], [6 : i32], [7 : i32], [8 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %unpacked_rhs = "iterators.map"(%four_to_seven) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)

  // Zip and print.
  %zipped = iterators.zip %unpacked_lhs, %unpacked_rhs :
                (!iterators.stream<i32>, !iterators.stream<i32>)
                  -> (!iterators.stream<tuple<i32, i32>>)
  "iterators.sink"(%zipped) : (!iterators.stream<tuple<i32, i32>>) -> ()

  // CHECK:      (0, 4)
  // CHECK-NEXT: (1, 5)
  // CHECK-NEXT: (2, 6)
  // CHECK-NEXT: (3, 7)
  return
}
