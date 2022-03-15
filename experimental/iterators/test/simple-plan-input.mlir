// RUN: mlir-proto-opt %s

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.iterator)
  %reduce = "iterators.reduce"(%input) : (!iterators.iterator) -> (!iterators.iterator)
  "iterators.sink"(%reduce) : (!iterators.iterator) -> ()
  return
}
