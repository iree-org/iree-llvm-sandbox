// RUN: mlir-proto-opt %s | FileCheck %s

// CHECK: linalg_transform.sequence
linalg_transform.sequence {
  // CHECK: tile "linalg.matmul" in @matmul_tensors {
  // CHECK_DAG: pad = false
  // CHECK-DAG: sizes = [4, 4, 4]
  // CHECK: }
  tile "linalg.matmul" in @matmul_tensors {sizes = [4, 4, 4], pad = false}
  // CHECK: decompose
  decompose
  // CHECK: vectorize "linalg.matmul" in @matmul_tensors {vectorize_padding = true}
  vectorize "linalg.matmul" in @matmul_tensors {vectorize_padding = true}
  // CHECK: bufferize
  bufferize
  // CHECK: lower_vectors {multireduction_lowering = "innerreduce"}
  lower_vectors { multireduction_lowering = "innerreduce"}
  // CHECK: lower_to_llvm
  lower_to_llvm
}
