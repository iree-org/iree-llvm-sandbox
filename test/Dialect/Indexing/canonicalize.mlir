// RUN: structured-opt -canonicalize -allow-unregistered-dialect -mlir-print-op-generic -split-input-file %s | FileCheck %s

// CHECK-LABEL: "builtin.module"() ({
// CHECK:         "func.func"() <{function_type = () -> (), sym_name = "test_canonicalize_to_attrs"}> ({
// CHECK:           %[[VAL_0:.*]] = "c0_index_dyn"() : () -> index
// CHECK:           %[[VAL_1:.*]] = "c100_index_dyn"() : () -> index
// CHECK:           %[[VAL_2:.*]] = "c2_index_dyn"() : () -> index
// CHECK:           %[[VAL_3:.*]] = "indexing.arange"(%[[VAL_0]]) {nofold, operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : index, stopAttr = 100 : index} : (index) -> tensor<?x1xindex>
// CHECK:           "use1"(%[[VAL_3]]) : (tensor<?x1xindex>) -> ()
// CHECK:           %[[VAL_4:.*]] = "indexing.arange"(%[[VAL_0]], %[[VAL_1]]) {nofold, operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : index} : (index, index) -> tensor<?x1xindex>
// CHECK:           "use2"(%[[VAL_4]]) : (tensor<?x1xindex>) -> ()
// CHECK:           %[[VAL_5:.*]] = "indexing.arange"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {nofold, operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
// CHECK:           "use3"(%[[VAL_5]]) : (tensor<?x1xindex>) -> ()
// CHECK:           %[[VAL_6:.*]] = "indexing.arange"(%[[VAL_0]]) {nofold, operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : index, stopAttr = 100 : index} : (index) -> tensor<?x1xindex>
// CHECK:           "use4"(%[[VAL_6]]) : (tensor<?x1xindex>) -> ()
// CHECK:           %[[VAL_7:.*]] = "indexing.arange"(%[[VAL_0]], %[[VAL_1]]) {nofold, operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : index} : (index, index) -> tensor<?x1xindex>
// CHECK:           "use5"(%[[VAL_7]]) : (tensor<?x1xindex>) -> ()
// CHECK:           "func.return"() : () -> ()
// CHECK:         }) : () -> ()
// CHECK:       }) : () -> ()

module {
  func.func @test_canonicalize_to_attrs() {
    %c100_index = arith.constant 100 : index
    %c2_index = arith.constant 2 : index
    %one = arith.constant 1 : index
    %c0_index_dyn = "c0_index_dyn"() : () -> (index)
    %c100_index_dyn = "c100_index_dyn"() : () -> (index)
    %c2_index_dyn = "c2_index_dyn"() : () -> (index)

    %10 = "indexing.arange"(%c0_index_dyn, %c100_index, %c2_index) {nofold, operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
    "use1"(%10) : (tensor<?x1xindex>) -> ()

    %11 = "indexing.arange"(%c0_index_dyn, %c100_index_dyn, %c2_index) {nofold, operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
    "use2"(%11) : (tensor<?x1xindex>) -> ()

    %12 = "indexing.arange"(%c0_index_dyn, %c100_index_dyn, %c2_index_dyn) {nofold, operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
    "use3"(%12) : (tensor<?x1xindex>) -> ()

    %13 = "indexing.arange"(%c0_index_dyn) {nofold, stopAttr = 100 : index, stepAttr = 2 : index, operand_segment_sizes = array<i32: 1, 0, 0>} : (index) -> tensor<?x1xindex>
    "use4"(%13) : (tensor<?x1xindex>) -> ()

    %14 = "indexing.arange"(%c0_index_dyn, %c100_index_dyn) {nofold, stepAttr = 2 : index, operand_segment_sizes = array<i32: 1, 1, 0>} : (index, index) -> tensor<?x1xindex>
    "use5"(%14) : (tensor<?x1xindex>) -> ()

    return
  }
}

// -----

// CHECK-LABEL: "builtin.module"() ({
// CHECK:         "func.func"() <{function_type = () -> (), sym_name = "test_fold"}> ({
// CHECK:           %[[VAL_0:.*]] = "arith.constant"() <{value = dense<{{\[\[}}0], [2], [4], [6], [8], [10], [12], [14], [16], [18], [20], [22], [24], [26], [28], [30], [32], [34], [36], [38], [40], [42], [44], [46], [48], [50], [52], [54], [56], [58], [60], [62], [64], [66], [68], [70], [72], [74], [76], [78], [80], [82], [84], [86], [88], [90], [92], [94], [96], [98]]> : tensor<50x1xindex>}> : () -> tensor<50x1xindex>
// CHECK:           "use1"(%[[VAL_0]]) : (tensor<50x1xindex>) -> ()
// CHECK:           "use2"(%[[VAL_0]]) : (tensor<50x1xindex>) -> ()
// CHECK:           %[[VAL_1:.*]] = "indexing.arange"() {nofold, operand_segment_sizes = array<i32: 0, 0, 0>, startAttr = 0 : index, stepAttr = 2 : index, stopAttr = 100 : index} : () -> tensor<50x1xindex>
// CHECK:           "use3"(%[[VAL_1]]) : (tensor<50x1xindex>) -> ()
// CHECK:           %[[VAL_2:.*]] = "indexing.arange"() {nofold, operand_segment_sizes = array<i32: 0, 0, 0>, startAttr = 0 : index, stepAttr = 2 : index, stopAttr = 100 : index} : () -> tensor<50x1xindex>
// CHECK:           "use4"(%[[VAL_2]]) : (tensor<50x1xindex>) -> ()
// CHECK:           "func.return"() : () -> ()
// CHECK:         }) : () -> ()
// CHECK:       }) : () -> ()

module {
  func.func @test_fold() {
    %c0_index = arith.constant 0 : index
    %c100_index = arith.constant 100 : index
    %c2_index = arith.constant 2 : index

    %4 = "indexing.arange"(%c0_index, %c100_index, %c2_index) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?x1xindex>
    "use1"(%4) : (tensor<?x1xindex>) -> ()
    %5 = "indexing.arange"(%c0_index, %c2_index) {stopAttr = 100 : index, operand_segment_sizes = array<i32: 1, 0, 1>} : (index, index) -> tensor<?x1xindex>
    "use2"(%5) : (tensor<?x1xindex>) -> ()
    %si = "indexing.arange"(%c0_index) {nofold, stopAttr = 100 : index, stepAttr = 2 : index, operand_segment_sizes = array<i32: 1, 0, 0>} : (index) -> tensor<?x1xindex>
    "use3"(%si) : (tensor<?x1xindex>) -> ()
    %se = "indexing.arange"() {nofold, startAttr = 0 : index, stopAttr = 100 : index, stepAttr = 2 : index, operand_segment_sizes = array<i32: 0, 0, 0>} : () -> tensor<50x1xindex>
    "use4"(%se) : (tensor<50x1xindex>) -> ()

    return
  }
}
