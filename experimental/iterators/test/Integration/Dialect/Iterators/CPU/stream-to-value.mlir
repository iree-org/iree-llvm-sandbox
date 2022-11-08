// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func @test_non_empty() {
  %i = arith.constant 42 : i32
  %undef = llvm.mlir.undef : !llvm.struct<(i32)>
  %value = llvm.insertvalue %i, %undef[0 : index] : !llvm.struct<(i32)>
  %stream = iterators.value_to_stream %value :
                !iterators.stream<!llvm.struct<(i32)>>
  %result:2 = iterators.stream_to_value %stream :
                  !iterators.stream<!llvm.struct<(i32)>>
  scf.if %result#1 {
    "iterators.print"(%result) : (!llvm.struct<(i32)>) -> ()
    // CHECK: (42)
  }
  return
}

func.func private @return_false(%struct : !llvm.struct<(i32)>) -> i1 {
  %false = arith.constant false
  return %false : i1
}

func.func @test_empty() {
  %i = arith.constant 1337 : i32
  %undef = llvm.mlir.undef : !llvm.struct<(i32)>
  %value = llvm.insertvalue %i, %undef[0 : index] : !llvm.struct<(i32)>
  %stream = iterators.value_to_stream %value :
                !iterators.stream<!llvm.struct<(i32)>>
  %filtered = "iterators.filter"(%stream) {predicateRef = @return_false} :
      (!iterators.stream<!llvm.struct<(i32)>>) ->
          (!iterators.stream<!llvm.struct<(i32)>>)
  %result:2 = iterators.stream_to_value %filtered :
                  !iterators.stream<!llvm.struct<(i32)>>
  // This shouldn't print anything.
  scf.if %result#1 {
    "iterators.print"(%result) : (!llvm.struct<(i32)>) -> ()
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
