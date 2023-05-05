// RUN: structured-opt %s \
// RUN: | FileCheck %s

// CHECK: module {
// CHECK:   func.func @foo(%arg0: !indexing.custom<"bob">) {
// CHECK:     %c2_i32 = arith.constant 2 : i32
// CHECK:     %0 = tensor.empty() : tensor<10x10xi32>
// CHECK:     %1 = tensor.empty() : tensor<1x2x2xi32>
// CHECK:     %2 = indexing.gather %0[%1] gather_dims([0, 1]) : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
// CHECK:     return
// CHECK:   }
// CHECK: }
module {
  func.func @foo(%arg0: !indexing.custom<"bob">) {
    %c2_i32 = arith.constant 2 : i32
    %0 = tensor.empty() : tensor<10x10xi32>
    %1 = tensor.empty() : tensor<1x2x2xi32>
    %2 = indexing.gather %0[%1] gather_dims([0, 1]) : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
    return
  }
}
