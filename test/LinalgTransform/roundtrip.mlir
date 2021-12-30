// RUN: mlir-proto-opt %s | FileCheck %s

// CHECK: linalg_transform.sequence
linalg_transform.sequence {
  // CHECK: %[[TILED:.*]] = tile when @match1 {
  // CHECK_DAG: pad = false
  // CHECK-DAG: sizes = [4, 4, 4]
  // CHECK: }
  %1 = tile when @match1 {sizes = [4, 4, 4], pad = false}
  // CHECK: %[[TILED2:.*]] = tile %[[TILED]]
  %2 = tile %1 {sizes = [2, 2, 2], pad = true}
  // CHECK: decompose
  decompose
  // CHECK: %{{.*}} = vectorize %[[TILED2]] {vectorize_padding = true}
  %3 = vectorize %2 {vectorize_padding = true}
  // CHECK: %{{.*}} = vectorize when @match2
  vectorize when @match2
  // CHECK: bufferize
  bufferize
  // CHECK: lower_vectors {multireduction_lowering = "innerreduce"}
  lower_vectors { multireduction_lowering = "innerreduce"}
  // CHECK: lower_to_llvm
  lower_to_llvm
}
