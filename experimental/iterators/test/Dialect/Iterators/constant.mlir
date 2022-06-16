// Test that we can parse and verify tuple literals without errors
// RUN: mlir-proto-opt %s

func.func @main() {
  %empty_tuple = "iterators.constanttuple"() { values = [] } : () -> tuple<>
  %one_field_tuple = "iterators.constanttuple"() { values = [1 : i32] } : () -> tuple<i32>
  %three_field_tuple = "iterators.constanttuple"() { values = [1 : i32, 2 : i32, 3 : i32] } : () -> tuple<i32, i32, i32>
  return
}
