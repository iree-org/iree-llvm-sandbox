// RUN: mlir-proto-opt %s -convert-iterators-to-llvm -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @print_empty_tuple(%tuple : tuple<>) -> () {
  "iterators.printtuple"(%tuple) : (tuple<>) -> ()
  return
}

func.func @main() {
  %empty_tuple = "iterators.constanttuple"() { values = [] } : () -> tuple<>
  "iterators.printtuple"(%empty_tuple) : (tuple<>) -> ()
  // CHECK:      ()

  func.call @print_empty_tuple(%empty_tuple) : (tuple<>) -> ()
  // CHECK:      ()

  %one_field_tuple = "iterators.constanttuple"() { values = [1 : i32] } : () -> tuple<i32>
  "iterators.printtuple"(%one_field_tuple) : (tuple<i32>) -> ()
  // CHECK-NEXT: (1)

  %three_field_tuple = "iterators.constanttuple"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  "iterators.printtuple"(%three_field_tuple) : (tuple<i32, i32, i32>) -> ()
  // CHECK-NEXT: (1, 2, 3)

  %mixed_field_tuple = "iterators.constanttuple"()
      { values = [1 : i1, 2 : i8, 3 : i16, 4 : i32, 5 : i64, 8.5 : f16, 8.25 : f32, 8.125 : f64] }
      : () -> tuple<i1, i8, i16, i32, i64, f16, f32, f64>
  "iterators.printtuple"(%mixed_field_tuple) : (tuple<i1, i8, i16, i32, i64, f16, f32, f64>) -> ()
  // CHECK-NEXT: (1, 2, 3, 4, 5, 8.5, 8.25, 8.125)

  %empty_struct = llvm.mlir.undef : !llvm.struct<()>
  "iterators.print"(%empty_struct) : (!llvm.struct<()>) -> ()
  // CHECK-NEXT: ()

  %two_field_struct = llvm.mlir.constant([1, 2]) : !llvm.struct<(i32, i32)>
  "iterators.print"(%two_field_struct) : (!llvm.struct<(i32, i32)>) -> ()
  // CHECK-NEXT: (1, 2)

  return
}
