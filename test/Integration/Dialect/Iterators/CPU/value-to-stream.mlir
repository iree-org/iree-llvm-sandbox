// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @main() {
  %value = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value :
                !iterators.stream<i32>
  "iterators.sink"(%stream) : (!iterators.stream<i32>) -> ()
  // CHECK: 42
  return
}
