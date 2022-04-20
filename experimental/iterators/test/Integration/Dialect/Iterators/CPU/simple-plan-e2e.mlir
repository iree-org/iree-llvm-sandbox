// RUN: mlir-proto-opt %s -convert-iterators-to-llvm -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libruntime_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.stream<tuple<i32>>)
  %reduce = "iterators.reduce"(%input) : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<tuple<i32>>)
  "iterators.sink"(%reduce) : (!iterators.stream<tuple<i32>>) -> ()
  // CHECK: (6)
  return
}
