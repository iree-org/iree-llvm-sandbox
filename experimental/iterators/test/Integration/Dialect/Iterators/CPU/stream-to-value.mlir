// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @test_non_empty() {
  %value = arith.constant 42 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
  %result:2 = iterators.stream_to_value %stream : !iterators.stream<i32>
  scf.if %result#1 {
    "iterators.print"(%result) : (i32) -> ()
    // CHECK: 42
  }
  return
}

func.func private @return_false(%arg : i32) -> i1 {
  %false = arith.constant false
  return %false : i1
}

func.func @test_empty() {
  %value = arith.constant 1337 : i32
  %stream = iterators.value_to_stream %value : !iterators.stream<i32>
  %filtered = "iterators.filter"(%stream) {predicateRef = @return_false} :
      (!iterators.stream<i32>) -> (!iterators.stream<i32>)
  %result:2 = iterators.stream_to_value %filtered : !iterators.stream<i32>
  // This shouldn't print anything.
  scf.if %result#1 {
    "iterators.print"(%result) : (i32) -> ()
  }
  // Check that next match is the given tuple to ensure that above print wasn't
  // executed.
  %tuple = "iterators.constanttuple"()
      { values = [12345 : i32] } : () -> tuple<i32>
  "iterators.printtuple"(%tuple) : (tuple<i32>) -> ()
  // CHECK-NEXT: (12345)
  return
}

func.func @main() {
  func.call @test_non_empty() : () -> ()
  func.call @test_empty() : () -> ()
  return
}
