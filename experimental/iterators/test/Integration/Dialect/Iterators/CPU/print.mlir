// RUN: mlir-proto-opt %s -convert-iterators-to-llvm -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func @main() {
  %emptyTuple = "iterators.constant"() { values = [] } : () -> tuple<>
  "iterators.print"(%emptyTuple) : (tuple<>) -> ()
  // CHECK:      ()

  %oneFieldTuple = "iterators.constant"() { values = [1 : i32] } : () -> tuple<i32>
  "iterators.print"(%oneFieldTuple) : (tuple<i32>) -> ()
  // CHECK-NEXT: (1)

  %threeFieldTuple = "iterators.constant"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  "iterators.print"(%threeFieldTuple) : (tuple<i32, i32, i32>) -> ()
  // CHECK-NEXT: (1, 2, 3)

  return
}
