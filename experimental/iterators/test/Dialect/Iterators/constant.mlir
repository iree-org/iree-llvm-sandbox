// Test that we can parse and verify tuple literals without errors
// RUN: mlir-proto-opt %s

func @main() {
  %emptyTuple = "iterators.constant"() { values = [] } : () -> tuple<>
  %oneFieldTuple = "iterators.constant"() { values = [1 : i32] } : () -> tuple<i32>
  %threeFieldTuple = "iterators.constant"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  return
}
