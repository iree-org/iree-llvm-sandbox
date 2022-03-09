// RUN: mlir-opt -allow-unregistered-dialect %s

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators<"it<i32>">)
  %reduce = "iterators.reduce"(%input) : (!iterators<"it<i32>">) -> (!iterators<"it<i32>">)
  "iterators.sink"(%reduce) : (!iterators<"it<i32>">) -> (!iterators<"it<i32>">)
  return
}
