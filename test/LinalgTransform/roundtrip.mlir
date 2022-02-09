// RUN: mlir-proto-opt %s | FileCheck %s

// CHECK: linalg_transform.sequence
linalg_transform.sequence {
  // CHECK: %[[OPS:.*]] = match @{{.*}}
  %0 = match @match1
  // CHECK: %[[TILED:.*]] = tile %[[OPS]] {
  // CHECK-DAG: pad = false
  // CHECK-DAG: sizes = [4, 4, 4]
  // CHECK: }
  %1 = tile %0 {sizes = [4, 4, 4], pad = false}
  // CHECK: %[[TILED2:.*]] = tile %[[TILED]]
  %2 = tile %1 {sizes = [2, 2, 2], pad = true}
  // CHECK: decompose
  decompose
  // CHECK: %{{.*}} = vectorize %[[TILED2]] {vectorize_padding = true}
  %3 = vectorize %2 {vectorize_padding = true}
  // CHECK: %[[OPS2:.*]] = match @{{.*}}
  %4 = match @match2
  // CHECK: %{{.*}} = vectorize %[[OPS2]]
  vectorize %4
  // CHECK: bufferize
  bufferize
  // CHECK: lower_vectors {multireduction_lowering = "innerreduce"}
  lower_vectors { multireduction_lowering = "innerreduce"}
  // CHECK: lower_to_llvm
  lower_to_llvm
}
