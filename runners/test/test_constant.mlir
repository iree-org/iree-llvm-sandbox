// RUN: mlir-opt %s -linalg-comprehensive-module-bufferize |\
// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-memref-to-llvm |\
// RUN: mlir-opt -convert-std-to-llvm -reconcile-unrealized-casts | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext | \
// RUN: tee | FileCheck %s --check-prefix=EXEC

func @main() {
  %c0 = constant 0: index
  %v0 = constant 0.0 : f32

  // Top of the function globals, memref.alloc, copy.
  %lhs = constant dense<[[2.]]> : tensor<1x1xf32>
  %rhs = constant dense<[[3.]]> : tensor<1x1xf32>
  %accum = constant dense<[[1.]]> : tensor<1x1xf32>

  %result_vector_0 = vector.transfer_read %lhs[%c0, %c0], %v0 : tensor<1x1xf32>, vector<1x1xf32>

  // EXEC: ( ( 2 ) )
  vector.print %result_vector_0: vector<1x1xf32>


  %result = linalg.matmul ins(%lhs, %rhs : tensor<1x1xf32>, tensor<1x1xf32>)
    outs(%accum: tensor<1x1xf32>) -> tensor<1x1xf32>

  %result_vector_1 = vector.transfer_read %result[%c0, %c0], %v0 : tensor<1x1xf32>, vector<1x1xf32>

  // EXEC: ( ( 7 ) )
  vector.print %result_vector_1: vector<1x1xf32>

  return
}
