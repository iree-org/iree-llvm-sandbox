// RUN: mlir-proto-opt %s \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -arith-bufferize -cse -convert-memref-to-llvm -reconcile-unrealized-casts \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!tuple_type = tuple<i32,i64>
!struct_type = !llvm.struct<(i32,i64)>

func.func @single_block() {
  %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %t2 = arith.constant dense<[3, 4, 5]> : tensor<3xi64>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  %m2 = bufferization.to_memref %t2 : memref<3xi64>
  %batch = "iterators.view_as_columnar_batch"(%m1, %m2)
    : (memref<3xi32>, memref<3xi64>) -> !iterators.columnar_batch<!tuple_type>
  %stream = "iterators.scan_columnar_batch"(%batch)
    : (!iterators.columnar_batch<!tuple_type>) -> !iterators.stream<!struct_type>
  "iterators.sink"(%stream) : (!iterators.stream<!struct_type>) -> ()
  // CHECK:      (0, 3)
  // CHECK-NEXT: (1, 4)
  // CHECK-NEXT: (2, 5)
  return
}

func.func @query(%batch : !iterators.columnar_batch<!tuple_type>) {
  %stream = "iterators.scan_columnar_batch"(%batch)
    : (!iterators.columnar_batch<!tuple_type>) -> !iterators.stream<!struct_type>
  "iterators.sink"(%stream) : (!iterators.stream<!struct_type>) -> ()
  return
}

func.func @function_arg() {
  %t1 = arith.constant dense<[9, 8, 7]> : tensor<3xi32>
  %t2 = arith.constant dense<[6, 5, 4]> : tensor<3xi64>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  %m2 = bufferization.to_memref %t2 : memref<3xi64>
  %batch = "iterators.view_as_columnar_batch"(%m1, %m2)
    : (memref<3xi32>, memref<3xi64>) -> !iterators.columnar_batch<!tuple_type>
  func.call @query(%batch) : (!iterators.columnar_batch<!tuple_type>) -> ()
  // CHECK-NEXT: (9, 6)
  // CHECK-NEXT: (8, 5)
  // CHECK-NEXT: (7, 4)
  return
}

func.func @main() {
  func.call @single_block() : () -> ()
  func.call @function_arg() : () -> ()
  return
}
