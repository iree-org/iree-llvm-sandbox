// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @main() {
  %i = arith.constant 42 : i32
  %undef = llvm.mlir.undef : !llvm.struct<(i32)>
  %value = llvm.insertvalue %i, %undef[0 : index] : !llvm.struct<(i32)>
  %stream = iterators.value_to_stream %value :
                !iterators.stream<!llvm.struct<(i32)>>
  "iterators.sink"(%stream) : (!iterators.stream<!llvm.struct<(i32)>>) -> ()
  // CHECK: (42)
  return
}
