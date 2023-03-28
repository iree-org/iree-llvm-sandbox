// RUN: iterators-opt %s | FileCheck %s

// RUN: iterators-opt %s | iterators-opt | FileCheck %s

// CHECK-LABEL: func.func @main(
// CHECK-SAME:                  %[[ARG0:.*]]: i64) {
func.func @main(%i32 : i64) {
  // CHECK-NEXT: iterators.print("prefix", %[[ARG0]], "suffix\0A") : i64
  iterators.print("prefix", %i32, "suffix\0A") : i64

  // CHECK-NEXT: iterators.print(%[[ARG0]], "suffix\0A") : i64
  iterators.print("", %i32, "suffix\0A") : i64

  // CHECK-NEXT: iterators.print(%[[ARG0]], "suffix\0A") : i64
  iterators.print(%i32, "suffix\0A") : i64

  // CHECK-NEXT: iterators.print("prefix", "suffix\0A")
  iterators.print("prefix", "suffix\0A")

  // CHECK-NEXT: iterators.print("prefix", %[[ARG0]]) : i64
  iterators.print("prefix", %i32, "\0A") : i64

  // CHECK-NEXT: iterators.print("prefix", %[[ARG0]]) : i64
  iterators.print("prefix", %i32) : i64

  // CHECK-NEXT: iterators.print("prefix")
  iterators.print("prefix", "\0A")

  // CHECK-NEXT: iterators.print("prefix")
  iterators.print("prefix")

  // CHECK-NEXT: iterators.print("", "suffix\0A")
  iterators.print("", "suffix\0A")

  // CHECK-NEXT: iterators.print("prefix", "")
  iterators.print("prefix", "")

  // CHECK-NEXT: iterators.print("\0A", "suffix\0A")
  iterators.print("\0A", "suffix\0A")

  // CHECK-NEXT: iterators.print(%[[ARG0]])
  iterators.print(%i32) : i64

  // CHECK-NEXT: iterators.print()
  iterators.print("")

  // CHECK-NEXT: iterators.print()
  iterators.print()

  return
}
