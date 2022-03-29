// RUN: mlir-proto-opt %s -convert-iterators-to-std -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libruntime_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %input = "iterators.sampleInput"() : () -> (!iterators.iterator<tuple<i32>>)
  %reduce = "iterators.reduce"(%input) : (!iterators.iterator<tuple<i32>>) -> (!iterators.iterator<tuple<i32>>)
  "iterators.sink"(%reduce) : (!iterators.iterator<tuple<i32>>) -> ()
  // CHECK: (6)
  return
}
