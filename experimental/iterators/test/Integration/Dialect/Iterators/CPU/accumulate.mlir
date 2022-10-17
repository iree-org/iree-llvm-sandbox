// RUN: iterators-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -decompose-tuples \
// RUN:   -inline -canonicalize \
// RUN:   -one-shot-bufferize="allow-return-allocs" \
// RUN:   -buffer-hoisting \
// RUN:   -buffer-deallocation \
// RUN:   -convert-bufferization-to-memref \
// RUN:   -expand-strided-metadata \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-scf-to-cf \
// RUN:   -convert-func-to-llvm \
// RUN:   -canonicalize \
// RUN:   -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner \
// RUN:   -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_lib_dir/libmlir_c_runner_utils%shlibext \
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

func.func private @unpack_i32(%input : tuple<i32>) -> i32 {
  %i = tuple.to_elements %input : tuple<i32>
  return %i : i32
}

func.func private @accumulate_histogram(
      %hist : tensor<4xi32>, %val : i32) -> tensor<4xi32> {
  %idx = arith.index_cast %val : i32 to index
  %oldCount = tensor.extract %hist[%idx] : tensor<4xi32>
  %one = arith.constant 1 : i32
  %newCount = arith.addi %oldCount, %one : i32
  %newHist = tensor.insert %newCount into %hist[%idx] : tensor<4xi32>
  return %newHist : tensor<4xi32>
}

func.func private @tensor_to_struct(%input : tensor<4xi32>) -> tuple<i32, i32, i32, i32> {
  %idx0 = arith.constant 0 : index
  %idx1 = arith.constant 1 : index
  %idx2 = arith.constant 2 : index
  %idx3 = arith.constant 3 : index
  %i0 = tensor.extract %input[%idx0] : tensor<4xi32>
  %i1 = tensor.extract %input[%idx1] : tensor<4xi32>
  %i2 = tensor.extract %input[%idx2] : tensor<4xi32>
  %i3 = tensor.extract %input[%idx3] : tensor<4xi32>
  %tuple = tuple.from_elements %i0, %i1, %i2, %i3 : tuple<i32, i32, i32, i32>
  return %tuple : tuple<i32, i32, i32, i32>
}

// CHECK-LABEL: test_accumulate_histogram
// CHECK-NEXT:  (1, 2, 1, 0)
// CHECK-NEXT:  -
func.func @test_accumulate_histogram() {
  iterators.print("test_accumulate_histogram")
  %input = "iterators.constantstream"()
      { value = [[0 : i32], [1 : i32], [1 : i32], [2 : i32]] }
      : () -> (!iterators.stream<tuple<i32>>)
  %unpacked = "iterators.map"(%input) {mapFuncRef = @unpack_i32}
    : (!iterators.stream<tuple<i32>>) -> (!iterators.stream<i32>)
  %init_value = arith.constant dense<[0, 0, 0, 0]> : tensor<4xi32>
  %accumulated = iterators.accumulate(%unpacked, %init_value)
    with @accumulate_histogram
      : (!iterators.stream<i32>) -> !iterators.stream<tensor<4xi32>>
  %transposed = "iterators.map"(%accumulated) {mapFuncRef = @tensor_to_struct}
    : (!iterators.stream<tensor<4xi32>>) -> (!iterators.stream<tuple<i32, i32, i32, i32>>)
  "iterators.sink"(%transposed) : (!iterators.stream<tuple<i32, i32, i32, i32>>) -> ()
  return
}

func.func @main() {
  call @test_accumulate_sum_tuple() : () -> ()
  call @test_accumulate_avg_tuple() : () -> ()
  call @test_accumulate_histogram() : () -> ()
  return
}
