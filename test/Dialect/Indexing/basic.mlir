// RUN: structured-opt %s --split-input-file | FileCheck %s

// CHECK: module {
// CHECK:   func.func @foo(%arg0: !indexing.custom<"bob">) {
// CHECK:     %[[TEN1:.*]] = tensor.empty() : tensor<10x10xi32>
// CHECK:     %[[IDXTEN:.*]] = tensor.empty() : tensor<1x2x2xi32>
// CHECK:     %[[GATHER:.*]] = indexing.gather %[[TEN1]][%[[IDXTEN]]] gather_dims([0, 1]) : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
// CHECK:     %[[CONCAT:.*]] = indexing.concatenate(%2, %2) {dim = 0} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
// CHECK:     %[[C0:.*]] = arith.constant 0 : i64
// CHECK:     %[[C100:.*]] = arith.constant 100 : i64
// CHECK:     %[[C2:.*]] = arith.constant 2 : i64
// CHECK:     %4 = indexing.arange(start = %[[C0]], stop = %[[C100]], step = %[[C2]]) : tensor<?xindex>
// CHECK:     return
// CHECK:   }
// CHECK: }
module {
  func.func @foo(%arg0: !indexing.custom<"bob">) {
    %0 = tensor.empty() : tensor<10x10xi32>
    %1 = tensor.empty() : tensor<1x2x2xi32>
    %2 = indexing.gather %0[%1] gather_dims([0, 1]) : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
    %3 = indexing.concatenate (%2, %2) {dim = 0} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c2_i64 = arith.constant 2 : i64
    %4 = indexing.arange(start = %c0_i64, stop = %c100_i64, step = %c2_i64) : tensor<?xindex>
    return
  }
}

// -----

// CHECK: module {
// CHECK:   %[[START:.*]] = arith.constant 0 : i64
// CHECK:   %[[STOP:.*]] = arith.constant 100 : i64
// CHECK:   %[[STEP:.*]] = arith.constant 2 : i64
// CHECK:   %{{.*}} = indexing.arange(start = %[[START]], stop = %[[STOP]], step = %[[STEP]]) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = 0, stop = %[[STOP]], step = %[[STEP]]) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = %[[START]], stop = 100, step = %[[STEP]]) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = %[[START]], stop = %[[STOP]], step = 2) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = 0, stop = 100, step = %[[STEP]]) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = 0, stop = %[[STOP]], step = 2) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = %[[START]], stop = 100, step = 2) : tensor<?xindex>
// CHECK:   %{{.*}} = indexing.arange(start = 0, stop = 100, step = 2) : tensor<50xindex>
// CHECK: }
module {
  %0 = arith.constant 0 : i64
  %1 = arith.constant 100 : i64
  %2 = arith.constant 2 : i64
  %3 = "indexing.arange"(%0, %1, %2) {operand_segment_sizes = array<i32: 1, 1, 1>} : (i64, i64, i64) -> tensor<?xindex>
  %4 = "indexing.arange"(%1, %2) {operand_segment_sizes = array<i32: 0, 1, 1>, startAttr = 0 : i64} : (i64, i64) -> tensor<?xindex>
  %5 = "indexing.arange"(%0, %2) {operand_segment_sizes = array<i32: 1, 0, 1>, stopAttr = 100 : i64} : (i64, i64) -> tensor<?xindex>
  %6 = "indexing.arange"(%0, %1) {operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : i64} : (i64, i64) -> tensor<?xindex>
  %7 = "indexing.arange"(%2) {operand_segment_sizes = array<i32: 0, 0, 1>, startAttr = 0 : i64, stopAttr = 100 : i64} : (i64) -> tensor<?xindex>
  %8 = "indexing.arange"(%1) {operand_segment_sizes = array<i32: 0, 1, 0>, startAttr = 0 : i64, stepAttr = 2 : i64} : (i64) -> tensor<?xindex>
  %9 = "indexing.arange"(%0) {operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : i64, stopAttr = 100 : i64} : (i64) -> tensor<?xindex>
  %10 = "indexing.arange"() {operand_segment_sizes = array<i32: 0, 0, 0>, startAttr = 0 : i64, stepAttr = 2 : i64, stopAttr = 100 : i64} : () -> tensor<50xindex>
}
