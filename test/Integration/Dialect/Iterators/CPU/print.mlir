// RUN: structured-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-tuples \
// RUN:   -cse -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @print_empty_tuple(%tuple : tuple<>) -> () {
  iterators.print(%tuple) : tuple<>
  return
}

func.func @main() {
  %i32 = arith.constant 42

  // CHECK:      constant
  iterators.print("constant")

  // CHECK-NEXT: 42
  iterators.print(%i32) : i64

  // CHECK-EMPTY:
  iterators.print("")

  // CHECK-EMPTY:
  iterators.print()

  // CHECK-NEXT: constant42
  iterators.print("constant" nonl)
  iterators.print("" nonl)
  iterators.print(nonl)
  iterators.print(%i32 nonl) : i64
  iterators.print()

  %empty_tuple = "iterators.constanttuple"() { values = [] } : () -> tuple<>
  iterators.print(%empty_tuple) : tuple<>
  // CHECK:      ()

  func.call @print_empty_tuple(%empty_tuple) : (tuple<>) -> ()
  // CHECK-NEXT: ()

  %one_field_tuple = "iterators.constanttuple"() { values = [1 : i32] } : () -> tuple<i32>
  iterators.print(%one_field_tuple) : tuple<i32>
  // CHECK-NEXT: (1)

  %three_field_tuple = "iterators.constanttuple"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  iterators.print(%three_field_tuple) : tuple<i32, i32, i32>
  // CHECK-NEXT: (1, 2, 3)

  %mixed_field_tuple = "iterators.constanttuple"()
      { values = [1 : i1, 2 : i8, 3 : i16, 4 : i32, 5 : i64, 8.5 : f16, 8.25 : f32, 8.125 : f64] }
      : () -> tuple<i1, i8, i16, i32, i64, f16, f32, f64>
  iterators.print(%mixed_field_tuple) : tuple<i1, i8, i16, i32, i64, f16, f32, f64>
  // CHECK-NEXT: (1, 2, 3, 4, 5, 8.5, 8.25, 8.125)

  %i1 = arith.constant false
  "iterators.print"(%i1) : (i1) -> ()
  // CHECK-NEXT: 0

  %i3 = arith.constant 0 : i3
  "iterators.print"(%i3) : (i3) -> ()
  // CHECK-NEXT: 0

  %complex = complex.constant [1.0 : f32, 0.0 : f32] : complex<f32>
  "iterators.print"(%complex) : (complex<f32>) -> ()
  // CHECK-NEXT: 1 + 0i

  %empty_struct = llvm.mlir.undef : !llvm.struct<()>
  "iterators.print"(%empty_struct) : (!llvm.struct<()>) -> ()
  // CHECK-NEXT: ()

  %empty_nested_struct = llvm.mlir.undef : !llvm.struct<(struct<()>)>
  "iterators.print"(%empty_nested_struct) : (!llvm.struct<(struct<()>)>) -> ()
  // CHECK-NEXT: (())

  %two_field_struct = llvm.mlir.constant([1, 2]) : !llvm.struct<(i32, i32)>
  "iterators.print"(%two_field_struct) : (!llvm.struct<(i32, i32)>) -> ()
  // CHECK-NEXT: (1, 2)

  return
}
