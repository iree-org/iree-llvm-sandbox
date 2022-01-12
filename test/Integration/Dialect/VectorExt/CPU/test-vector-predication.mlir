// RUN: mlir-proto-opt %s -test-vector-masking-utils=predication -test-vector-masking-utils=masking \
// RUN:   -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm -convert-std-to-llvm \
// RUN:   -reconcile-unrealized-casts | \
// RUN: mlir-cpu-runner -e entry -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils_dir/libmlir_c_runner_utils%shlibext | \
// RUN: FileCheck %s

func @func_pred0(%arg0: memref<8xf32>, %pred0: vector<8xi1>) {
  %c0 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %0 = vector.transfer_read %arg0[%c0], %f0 {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  %add = arith.addf %0, %0 : vector<8xf32>
  vector.transfer_write %add, %arg0[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>
  return
}

func @entry() {
  %f0 = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %buf = memref.alloca() : memref<8xf32>
  %init_val = arith.constant dense<3.0> : vector<8xf32>
  vector.transfer_write %init_val, %buf[%c0] {in_bounds = [true]} : vector<8xf32>, memref<8xf32>

  %mask = arith.constant dense<[0, 0, 1, 1, 1, 1, 0, 0]> : vector<8xi1>
  call @func_pred0(%buf, %mask) : (memref<8xf32>, vector<8xi1>) -> ()

  %res = vector.transfer_read %buf[%c0], %f0 {in_bounds = [true]} : memref<8xf32>, vector<8xf32>
  vector.print %res : vector<8xf32>
  return
}

// CHECK: ( 3, 3, 6, 6, 6, 6, 3, 3 )
