// RUN: mlir-proto-opt %s \
// RUN:   -convert-tabular-to-llvm \
// RUN:   -convert-iterators-to-llvm \
// RUN:   -convert-states-to-llvm \
// RUN:   -arith-bufferize -cse -convert-memref-to-llvm -reconcile-unrealized-casts \
// RUN:   -convert-func-to-llvm \
// RUN:   -convert-scf-to-cf -convert-cf-to-llvm \
// RUN: | mlir-cpu-runner -e main -entry-point-result=void \
// RUN: | FileCheck %s

!struct_type = !llvm.struct<(i32,i64)>

func.func @single_block() {
  %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %t2 = arith.constant dense<[3, 4, 5]> : tensor<3xi64>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  %m2 = bufferization.to_memref %t2 : memref<3xi64>
  %view = "tabular.view_as_tabular"(%m1, %m2)
    : (memref<3xi32>, memref<3xi64>) -> !tabular.tabular_view<i32,i64>
  %stream = iterators.tabular_view_to_stream %view
    to !iterators.stream<!struct_type>
  "iterators.sink"(%stream) : (!iterators.stream<!struct_type>) -> ()
  // CHECK:      (0, 3)
  // CHECK-NEXT: (1, 4)
  // CHECK-NEXT: (2, 5)
  return
}

func.func @query(%view : !tabular.tabular_view<i32,i64>) {
  %stream = iterators.tabular_view_to_stream %view
    to !iterators.stream<!struct_type>
  "iterators.sink"(%stream) : (!iterators.stream<!struct_type>) -> ()
  return
}

func.func @function_arg() {
  %t1 = arith.constant dense<[9, 8, 7]> : tensor<3xi32>
  %t2 = arith.constant dense<[6, 5, 4]> : tensor<3xi64>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  %m2 = bufferization.to_memref %t2 : memref<3xi64>
  %view = "tabular.view_as_tabular"(%m1, %m2)
    : (memref<3xi32>, memref<3xi64>) -> !tabular.tabular_view<i32,i64>
  func.call @query(%view) : (!tabular.tabular_view<i32,i64>) -> ()
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
