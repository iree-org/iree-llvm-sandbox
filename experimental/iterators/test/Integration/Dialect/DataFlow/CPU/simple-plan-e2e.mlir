// RUN: mlir-proto-opt %s -convert-dataflow-to-iterators -convert-func-to-llvm | \
// RUN: mlir-cpu-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libruntime_utils%shlibext \
// RUN: | FileCheck %s

func @main() {
  %input = "dataflow.sampleInput"() : () -> (!dataflow.stream<tuple<i32>>)
  %reduce = "dataflow.reduce"(%input) : (!dataflow.stream<tuple<i32>>) -> (!dataflow.stream<tuple<i32>>)
  "dataflow.sink"(%reduce) : (!dataflow.stream<tuple<i32>>) -> ()
  // CHECK: (6)
  return
}
