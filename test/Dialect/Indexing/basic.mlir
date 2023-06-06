// RUN: structured-opt %s --split-input-file | FileCheck %s

// CHECK-LABEL:   module {
// CHECK:           func.func @foo(%[[VAL_0:.*]]: !indexing.custom<"bob">) {
// CHECK:             %[[VAL_1:.*]] = tensor.empty() : tensor<10x10xi32>
// CHECK:             %[[VAL_2:.*]] = tensor.empty() : tensor<1x2x2xi32>
// CHECK:             %[[VAL_3:.*]] = indexing.gather %[[VAL_1]]{{\[}}%[[VAL_2]]] gather_dims([0, 1]) : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
// CHECK:             %[[VAL_4:.*]] = indexing.concatenate(%[[VAL_3]], %[[VAL_3]]) {dim = 0} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
// CHECK:             %[[VAL_5:.*]] = arith.constant 0 : index
// CHECK:             %[[VAL_6:.*]] = arith.constant 100 : index
// CHECK:             %[[VAL_7:.*]] = arith.constant 2 : index
// CHECK:             %[[VAL_8:.*]] = indexing.arange(start = %[[VAL_5]], stop = %[[VAL_6]], step = %[[VAL_7]]) : tensor<?xindex>
// CHECK:             return
// CHECK:           }
// CHECK:         }

module {
  func.func @foo(%arg0: !indexing.custom<"bob">) {
    %0 = tensor.empty() : tensor<10x10xi32>
    %1 = tensor.empty() : tensor<1x2x2xi32>
    %2 = indexing.gather %0[%1] gather_dims([0, 1]) : (tensor<10x10xi32>, tensor<1x2x2xi32>) -> tensor<1x2xi32>
    %3 = indexing.concatenate (%2, %2) {dim = 0} : (tensor<1x2xi32>, tensor<1x2xi32>) -> tensor<2x2xi32>
    %c0_index = arith.constant 0 : index
    %c100_index = arith.constant 100 : index
    %c2_index = arith.constant 2 : index
    %4 = indexing.arange(start = %c0_index, stop = %c100_index, step = %c2_index) : tensor<?xindex>
    return
  }
}

// -----

// CHECK-LABEL:   module {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = indexing.arange(start = %[[VAL_0]], stop = %[[VAL_1]], step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_4:.*]] = indexing.arange(start = 0, stop = %[[VAL_1]], step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_5:.*]] = indexing.arange(start = %[[VAL_0]], stop = 100, step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_6:.*]] = indexing.arange(start = %[[VAL_0]], stop = %[[VAL_1]], step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_7:.*]] = indexing.arange(start = 0, stop = 100, step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_8:.*]] = indexing.arange(start = 0, stop = %[[VAL_1]], step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_9:.*]] = indexing.arange(start = %[[VAL_0]], stop = 100, step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_10:.*]] = indexing.arange(start = 0, stop = 100, step = 2) : tensor<50xindex>
// CHECK:         }

module {
  %0 = arith.constant 0 : index
  %1 = arith.constant 100 : index
  %2 = arith.constant 2 : index
  %3 = "indexing.arange"(%0, %1, %2) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?xindex>
  %4 = "indexing.arange"(%1, %2) {operand_segment_sizes = array<i32: 0, 1, 1>, startAttr = 0 : index} : (index, index) -> tensor<?xindex>
  %5 = "indexing.arange"(%0, %2) {operand_segment_sizes = array<i32: 1, 0, 1>, stopAttr = 100 : index} : (index, index) -> tensor<?xindex>
  %6 = "indexing.arange"(%0, %1) {operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : index} : (index, index) -> tensor<?xindex>
  %7 = "indexing.arange"(%2) {operand_segment_sizes = array<i32: 0, 0, 1>, startAttr = 0 : index, stopAttr = 100 : index} : (index) -> tensor<?xindex>
  %8 = "indexing.arange"(%1) {operand_segment_sizes = array<i32: 0, 1, 0>, startAttr = 0 : index, stepAttr = 2 : index} : (index) -> tensor<?xindex>
  %9 = "indexing.arange"(%0) {operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : index, stopAttr = 100 : index} : (index) -> tensor<?xindex>
  %10 = "indexing.arange"() {operand_segment_sizes = array<i32: 0, 0, 0>, startAttr = 0 : index, stepAttr = 2 : index, stopAttr = 100 : index} : () -> tensor<50xindex>
}

// -----

// CHECK-LABEL:   module {
// CHECK:           %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_1:.*]] = arith.constant 100 : index
// CHECK:           %[[VAL_2:.*]] = arith.constant 2 : index
// CHECK:           %[[VAL_3:.*]] = indexing.arange(start = %[[VAL_0]], stop = %[[VAL_1]], step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_4:.*]] = indexing.arange(start = 0, stop = %[[VAL_1]], step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_5:.*]] = indexing.arange(start = %[[VAL_0]], stop = 100, step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_6:.*]] = indexing.arange(start = %[[VAL_0]], stop = %[[VAL_1]], step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_7:.*]] = indexing.arange(start = 0, stop = 100, step = %[[VAL_2]]) : tensor<?xindex>
// CHECK:           %[[VAL_8:.*]] = indexing.arange(start = 0, stop = %[[VAL_1]], step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_9:.*]] = indexing.arange(start = %[[VAL_0]], stop = 100, step = 2) : tensor<?xindex>
// CHECK:           %[[VAL_10:.*]] = indexing.arange(start = 0, stop = 100, step = 2) : tensor<50xindex>
// CHECK:         }

module {
  %c0 = arith.constant 0 : index
  %c100 = arith.constant 100 : index
  %c2 = arith.constant 2 : index
  %0 = indexing.arange(start = %c0, stop = %c100, step = %c2) : tensor<?xindex>
  %1 = indexing.arange(start = 0, stop = %c100, step = %c2) : tensor<?xindex>
  %2 = indexing.arange(start = %c0, stop = 100, step = %c2) : tensor<?xindex>
  %3 = indexing.arange(start = %c0, stop = %c100, step = 2) : tensor<?xindex>
  %4 = indexing.arange(start = 0, stop = 100, step = %c2) : tensor<?xindex>
  %5 = indexing.arange(start = 0, stop = %c100, step = 2) : tensor<?xindex>
  %6 = indexing.arange(start = %c0, stop = 100, step = 2) : tensor<?xindex>
  %7 = indexing.arange(start = 0, stop = 100, step = 2) : tensor<50xindex>
}
