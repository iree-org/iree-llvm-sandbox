// Test that we can parse and verify constant streams without errors
// RUN: mlir-proto-opt %s

func.func @main() {
  %empty = "iterators.constantstream"() { value = [] }
      : () -> (!iterators.stream<!llvm.struct<(i32)>>)
  %i32 = "iterators.constantstream"() { value = [[42 : i32]] }
      : () -> (!iterators.stream<!llvm.struct<(i32)>>)
  %f32 = "iterators.constantstream"() { value = [[42. : f32]] }
      : () -> (!iterators.stream<!llvm.struct<(f32)>>)
  %i32i64 = "iterators.constantstream"() { value = [[42 : i32, 1337 : i64]] }
      : () -> (!iterators.stream<!llvm.struct<(i32, i64)>>)
  return
}
