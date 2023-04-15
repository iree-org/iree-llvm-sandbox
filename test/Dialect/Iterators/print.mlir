// RUN: structured-opt %s | FileCheck %s

// RUN: structured-opt %s | structured-opt | FileCheck %s

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %[[ARG0:.*]]: i64) {
func.func @main(%i32 : i64) {
  // CHECK-NEXT: iterators.print("constant")
  iterators.print("constant")

  // CHECK-NEXT: iterators.print( %[[ARG0]]) : i64
  iterators.print(%i32) : i64

  // CHECK-NEXT: iterators.print( )
  iterators.print("")

  // CHECK-NEXT: iterators.print( )
  iterators.print()

  // CHECK-NEXT: iterators.print("constant" nonl)
  iterators.print("constant" nonl)

  // CHECK-NEXT: iterators.print( %[[ARG0]] nonl) : i64
  iterators.print(%i32 nonl) : i64

  // CHECK-NEXT: iterators.print( nonl)
  iterators.print("" nonl)

  // CHECK-NEXT: iterators.print( nonl)
  iterators.print(nonl)

  return
}
