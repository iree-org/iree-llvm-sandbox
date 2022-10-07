// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @main() {
  %zero_to_three = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [2 : i32], [3 : i32]] }
      : () -> (!iterators.stream<!llvm.struct<(i32)>>)
  "iterators.sink"(%zero_to_three) : (!iterators.stream<!llvm.struct<(i32)>>) -> ()
  // CHECK:      (0)
  // CHECK-NEXT: (1)
  // CHECK-NEXT: (2)
  // CHECK-NEXT: (3)

  %empty = "iterators.constantstream"() { value = [] }
      : () -> (!iterators.stream<!llvm.struct<(i32)>>)
  "iterators.sink"(%empty) : (!iterators.stream<!llvm.struct<(i32)>>) -> ()
  // This should not produce any line, which the next CHECK-NEXT ensures.

  %mixed_types = "iterators.constantstream"()
      { value = [[1 : i1, 2 : i8, 3 : i16, 4 : i32, 5 : i64, 8.5 : f16, 8.25 : f32, 8.125 : f64]] }
      : () -> (!iterators.stream<!llvm.struct<(i1, i8, i16, i32, i64, f16, f32, f64)>>)
  "iterators.sink"(%mixed_types) : (!iterators.stream<!llvm.struct<(i1, i8, i16, i32, i64, f16, f32, f64)>>) -> ()
  // CHECK-NEXT: (1, 2, 3, 4, 5, 8.5, 8.25, 8.125)

  return
}
