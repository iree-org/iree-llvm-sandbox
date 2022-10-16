// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -decompose-tuples \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

func.func private @accumulate_sum_tuple(
      %acc : tuple<i32>, %val : tuple<i32>) -> tuple<i32> {
  %acci = tuple.to_elements %acc : tuple<i32>
  %vali = tuple.to_elements %val : tuple<i32>
  %i = arith.addi %acci, %vali : i32
  %result = tuple.from_elements %i : tuple<i32>
  return %result : tuple<i32>
}

// CHECK-LABEL: test_accumulate_sum_tuple
// CHECK-NEXT:  (160)
// CHECK-NEXT:  -
func.func @test_accumulate_sum_tuple() {
  iterators.print("test_accumulate_sum_tuple")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [10 : i32], [20 : i32], [30 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %hundred = arith.constant 100 : i32
  %init_value = tuple.from_elements %hundred : tuple<i32>
  %accumulated = iterators.accumulate(%input, %init_value)
    with @accumulate_sum_tuple
      : (!iterators.stream<tuple<i32>>) -> !iterators.stream<tuple<i32>>
  "iterators.sink"(%accumulated) : (!iterators.stream<tuple<i32>>) -> ()
  return
}

func.func private @accumulate_avg_tuple(
      %acc : tuple<i32, i32>, %val : tuple<i32>) -> tuple<i32, i32> {
  %cnt, %sum = tuple.to_elements %acc : tuple<i32, i32>
  %vali = tuple.to_elements %val : tuple<i32>
  %one = arith.constant 1 : i32
  %new_cnt = arith.addi %cnt, %one : i32
  %new_sum = arith.addi %sum, %vali : i32
  %result = tuple.from_elements %new_cnt, %new_sum : tuple<i32, i32>
  return %result : tuple<i32, i32>
}

func.func private @avg(%input : tuple<i32, i32>) -> tuple<f32> {
  %cnt, %sum = tuple.to_elements %input : tuple<i32, i32>
  %cntf = arith.sitofp %cnt : i32 to f32
  %sumf = arith.sitofp %sum : i32 to f32
  %avg = arith.divf %sumf, %cntf : f32
  %result = tuple.from_elements %avg : tuple<f32>
  return %result : tuple<f32>
}

// CHECK-LABEL: test_accumulate_avg_tuple
// CHECK-NEXT:  (15)
// CHECK-NEXT:  -
func.func @test_accumulate_avg_tuple() {
  iterators.print("test_accumulate_avg_tuple")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [10 : i32], [20 : i32], [30 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %zero = arith.constant 0 : i32
  %init_value = tuple.from_elements %zero, %zero : tuple<i32, i32>
  %accumulated = iterators.accumulate(%input, %init_value)
    with @accumulate_avg_tuple
      : (!iterators.stream<tuple<i32>>) -> !iterators.stream<tuple<i32, i32>>
  %mapped = "iterators.map"(%accumulated) {mapFuncRef = @avg}
    : (!iterators.stream<tuple<i32, i32>>) -> (!iterators.stream<tuple<f32>>)
  "iterators.sink"(%mapped) : (!iterators.stream<tuple<f32>>) -> ()
  return
}

func.func @main() {
  call @test_accumulate_sum_tuple() : () -> ()
  call @test_accumulate_avg_tuple() : () -> ()
  return
}
