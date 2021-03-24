// Check that inplace bufferization works with 2-level tiling + innermost padding + hoisting.
// RUN: mlir-proto-opt %s -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=8" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=4" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=2 pad hoist-padding=2" |\
// TODO: fix vectorization bug and enable.
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot vectorize vector-contract-lowering=false vectorize-padding" |\
// RUN: mlir-opt -canonicalize -cse |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\

// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext |\
// RUN: tee | FileCheck %s

// Check that inplace bufferization works with 3-level tiling + innermost padding + hoisting.
// RUN: mlir-proto-opt %s -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=8" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=4" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=2 pad hoist-padding=3" |\
// TODO: fix vectorization bug and enable.
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot vectorize vector-contract-lowering=false vectorize-padding" |\
// RUN: mlir-opt -canonicalize -cse |\
// RUN: mlir-proto-opt -linalg-comprehensive-bufferize-inplace |\

// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops |\
// RUN: mlir-opt -canonicalize -convert-scf-to-std -convert-vector-to-llvm -convert-std-to-llvm | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext |\
// RUN: tee | FileCheck %s

func @init_and_dot(%a: tensor<?xf32>, %b: tensor<?xf32>, %c: tensor<f32>) -> tensor<f32>
// TODO: activate manually for now.
// attributes { passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]]}
//
// Manually set up `__inplace_attr__` to allow writing tests in the absence of
// an external function call.
attributes { __inplace_args_attr__ = ["none", "none", "true"] }
{
  %v0 = constant 0.0 : f32

  %d = linalg.fill(%c, %v0) : tensor<f32>, f32 -> tensor<f32>

  %e = linalg.dot ins(%a, %b : tensor<?xf32>,tensor<?xf32>)
    outs(%d: tensor<f32>) -> tensor<f32>

  return %e : tensor<f32>
}

func @main() {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32


  %c13 = constant 13: index

  %A = linalg.init_tensor [%c13] : tensor<?xf32>
  %B = linalg.init_tensor [%c13] : tensor<?xf32>
  %C = linalg.init_tensor [] : tensor<f32>
  %AA = linalg.fill(%A, %v1) : tensor<?xf32>, f32 -> tensor<?xf32>
  %BB = linalg.fill(%B, %v2) : tensor<?xf32>, f32 -> tensor<?xf32>
  %CC = linalg.fill(%C, %v0) : tensor<f32>, f32 -> tensor<f32>

  %c0 = constant 0: index

  %res = linalg.dot ins(%AA, %BB : tensor<?xf32>,tensor<?xf32>)
                   outs(%CC: tensor<f32>) -> tensor<f32>

  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

//      CHECK: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// CHECK-NEXT: [26]
  call @print_memref_f32(%res2) : (tensor<*xf32>) -> ()

  return
}

// Abuse reliance on conversions by allowing `tensor<*xf32>`.
func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
