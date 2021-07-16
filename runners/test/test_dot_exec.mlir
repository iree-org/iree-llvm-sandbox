// Check that inplace bufferization works with 2-level tiling + innermost padding + hoisting.
// RUN: mlir-proto-opt %s -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=20" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=10" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=2 pad hoist-padding=2" |\

// TODO: Vectorizing linalg.dot requires 0-D vectors, disable for now.
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot vectorize vector-contract-lowering=false vectorize-padding" |\

// RUN: mlir-opt -canonicalize -cse |\
// RUN: mlir-opt -linalg-comprehensive-module-bufferize |\

// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops -convert-scf-to-std |\
// RUN: mlir-opt -convert-memref-to-llvm -convert-vector-to-llvm -convert-std-to-llvm -canonicalize -cse | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext |\
// RUN: tee | FileCheck %s

// Check that inplace bufferization works with 3-level tiling + innermost padding + hoisting.
// RUN: mlir-proto-opt %s -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=20" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=10" |\
// RUN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot tile-sizes=2 pad hoist-padding=3" |\

// TODO: fix vectorization bug and enable.
// R-UN: mlir-proto-opt -linalg-tensor-codegen-strategy="anchor-func=init_and_dot anchor-op=linalg.dot vectorize vector-contract-lowering=false vectorize-padding" |\

// RUN: mlir-opt -canonicalize -cse -linalg-comprehensive-module-bufferize |\
// RUN: mlir-opt -convert-vector-to-scf -lower-affine -convert-linalg-to-loops -convert-scf-to-std |\
// RUN: mlir-opt -convert-memref-to-llvm -convert-vector-to-llvm -convert-std-to-llvm -canonicalize -cse | \

// RUN: mlir-cpu-runner -O3 -e main -entry-point-result=void \
// RUN:   -shared-libs=%iree_runners_test_dir/libruntime-support%shlibext |\
// RUN: tee | FileCheck %s

func @init_and_dot(%a: tensor<64xf32>, %b: tensor<64xf32>, %c: tensor<f32>) -> tensor<f32> {
  %v0 = constant 0.0 : f32

  %d = linalg.fill(%v0, %c) : f32, tensor<f32> -> tensor<f32>

  %e = linalg.dot ins(%a, %b : tensor<64xf32>,tensor<64xf32>)
    outs(%d: tensor<f32>) -> tensor<f32>

  return %e : tensor<f32>
}


func @main() {
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %A = linalg.init_tensor [64] : tensor<64xf32>
  %B = linalg.init_tensor [64] : tensor<64xf32>
  %C = linalg.init_tensor [] : tensor<f32>
  %AA = linalg.fill(%v1, %A) : f32, tensor<64xf32> -> tensor<64xf32>
  %BB = linalg.fill(%v2, %B) : f32, tensor<64xf32> -> tensor<64xf32>
  %CC = linalg.fill(%v0, %C) : f32, tensor<f32> -> tensor<f32>

  %c0 = constant 0: index

  %res = call @init_and_dot(%AA, %BB, %CC) :
    (tensor<64xf32>, tensor<64xf32>, tensor<f32>) -> tensor<f32>

  %res2 = tensor.cast %res: tensor<f32> to tensor<*xf32>

//      CHECK: Unranked Memref base@ = {{.*}} rank = 0 offset = 0 sizes = [] strides = [] data =
// CHECK-NEXT: [128]
  call @print_memref_f32(%res2) : (tensor<*xf32>) -> ()

  return
}

func private @print_memref_f32(tensor<*xf32>) attributes { llvm.emit_c_interface }
