// RUN: iterators-opt %s \
// RUN:   -sparse-buffer-rewrite \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -decompose-iterator-states \
// RUN:   -decompose-tuples \
// RUN:   -convert-tabular-to-llvm \
// RUN:   -inline -canonicalize \
// RUN:   -arith-bufferize \
// RUN:   -convert-scf-to-cf \
// RUN:   -expand-strided-metadata \
// RUN:   -finalize-memref-to-llvm \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-cf-to-llvm \
// RUN:   -reconcile-unrealized-casts \
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

!memref_i32 = memref<?xi32>

func.func private @accumulate_realloc(
      %acc : !memref_i32, %val : tuple<i32>) -> !memref_i32 {
  %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %dim = memref.dim %acc, %zero : !memref_i32
  %new_dim = arith.addi %one, %dim : index
  %realloced = memref.realloc %acc (%new_dim) : !memref_i32 to !memref_i32
  %vali = tuple.to_elements %val : tuple<i32>
  memref.store %vali, %realloced[%dim] : !memref_i32
  return %realloced : !memref_i32
}

// CHECK-LABEL: test_accumulate_realloc
// CHECK-NEXT:  (9)
// CHECK-NEXT:  (8)
// CHECK-NEXT:  (7)
// CHECK-NEXT:  -
func.func @test_accumulate_realloc() {
  iterators.print("test_accumulate_realloc")
  %tensor = arith.constant dense<[9, 8, 7]> : tensor<3xi32>
  %memref = bufferization.to_memref %tensor : memref<3xi32>
  %view = "tabular.view_as_tabular"(%memref)
    : (memref<3xi32>) -> !tabular.tabular_view<i32>
  %stream = iterators.tabular_view_to_stream %view
    to !iterators.stream<tuple<i32>>
  %zero = arith.constant 0 : index
  %alloced = memref.alloc (%zero) : !memref_i32
  %accumulated = iterators.accumulate(%stream, %alloced)
    with @accumulate_realloc
      : (!iterators.stream<tuple<i32>>) -> !iterators.stream<!memref_i32>
  %result:2 = iterators.stream_to_value %accumulated :
                  !iterators.stream<!memref_i32>
  scf.if %result#1 {
    %result_view = "tabular.view_as_tabular"(%result#0)
      : (memref<?xi32>) -> !tabular.tabular_view<i32>
    %result_stream = iterators.tabular_view_to_stream %result_view
      to !iterators.stream<tuple<i32>>
    "iterators.sink"(%result_stream) : (!iterators.stream<tuple<i32>>) -> ()
  }
  return
}

func.func private @accumulate_push_back(
      %acc : tuple<index, !memref_i32>, %val : tuple<i32>) -> tuple<index, !memref_i32> {
  %size, %memref = tuple.to_elements %acc : tuple<index, !memref_i32>
  %vali = tuple.to_elements %val : tuple<i32>
  %new_memref, %new_size = sparse_tensor.push_back %size, %memref, %vali
   : index, memref<?xi32>, i32
  %result = tuple.from_elements %new_size, %new_memref : tuple<index, !memref_i32>
  return %result : tuple<index, !memref_i32>
}

// CHECK-LABEL: test_accumulate_push_back
// CHECK-NEXT:  (11)
// CHECK-NEXT:  (12)
// CHECK-NEXT:  (13)
// CHECK-NEXT:  -
func.func @test_accumulate_push_back() {
  iterators.print("test_accumulate_push_back")
  %tensor = arith.constant dense<[11, 12, 13]> : tensor<3xi32>
  %memref = bufferization.to_memref %tensor : memref<3xi32>
  %view = "tabular.view_as_tabular"(%memref)
    : (memref<3xi32>) -> !tabular.tabular_view<i32>
  %stream = iterators.tabular_view_to_stream %view
    to !iterators.stream<tuple<i32>>
    %zero = arith.constant 0 : index
  %one = arith.constant 1 : index
  %alloced = memref.alloc (%one) : !memref_i32
  %init_value = tuple.from_elements %zero, %alloced : tuple<index, !memref_i32>
  %accumulated = iterators.accumulate(%stream, %init_value)
    with @accumulate_push_back
      : (!iterators.stream<tuple<i32>>)
        -> !iterators.stream<tuple<index, !memref_i32>>
  %result:2 = iterators.stream_to_value %accumulated :
                  !iterators.stream<tuple<index, !memref_i32>>
  scf.if %result#1 {
    %result_size, %result_memref = tuple.to_elements %result#0 : tuple<index, !memref_i32>
    %final_memref = memref.realloc %result_memref (%result_size) : !memref_i32 to !memref_i32
    %result_view = "tabular.view_as_tabular"(%final_memref)
      : (memref<?xi32>) -> !tabular.tabular_view<i32>
    %result_stream = iterators.tabular_view_to_stream %result_view
      to !iterators.stream<tuple<i32>>
    "iterators.sink"(%result_stream) : (!iterators.stream<tuple<i32>>) -> ()
    memref.dealloc %final_memref : !memref_i32
  }
  return
}

func.func @main() {
  call @test_accumulate_sum_tuple() : () -> ()
  call @test_accumulate_avg_tuple() : () -> ()
  call @test_accumulate_realloc() : () -> ()
  call @test_accumulate_push_back() : () -> ()
  return
}
