// RUN: mlir-proto-opt %s

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.iterator<tuple<i32>>)
  %reduce = "iterators.reduce"(%input) : (!iterators.iterator<tuple<i32>>) -> (!iterators.iterator<tuple<i32>>)
  "iterators.sink"(%reduce) : (!iterators.iterator<tuple<i32>>) -> ()
  return
}
