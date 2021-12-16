// RUN: mlir-proto-opt %s | FileCheck %s

// CHECK: linalg_transform.apply
linalg_transform.apply {
  // CHECK: linalg_transform.sequence
  linalg_transform.sequence {
  // CHECK: ^{{.*}}(%[[ARG0:.*]]: !pdl.operation)
  ^bb0(%arg0: !pdl.operation):
    // CHECK: %[[TILED:.*]] = tile %[[ARG0]] {
    // CHECK_DAG: pad = false
    // CHECK-DAG: sizes = [4, 4, 4]
    // CHECK: }
    %1 = tile %arg0 {sizes = [4, 4, 4], pad = false}
    // CHECK: decompose
    decompose
    // CHECK: %{{.*}} = vectorize %[[TILED]] {vectorize_padding = true}
    %2 = vectorize %1 {vectorize_padding = true}
    // CHECK: bufferize
    bufferize
    // CHECK: lower_vectors {multireduction_lowering = "innerreduce"}
    lower_vectors { multireduction_lowering = "innerreduce"}
    // CHECK: lower_to_llvm
    lower_to_llvm
  }
// CHECK: when
} when {
  // CHECK: linalg_transform.pdl_match @match
  linalg_transform.pdl_match @match
}
