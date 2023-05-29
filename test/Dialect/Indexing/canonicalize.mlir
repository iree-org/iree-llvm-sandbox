// RUN: structured-opt -canonicalize -allow-unregistered-dialect -mlir-print-op-generic -split-input-file %s | FileCheck %s

// CHECK-LABEL:   "builtin.module"() ({
// CHECK:           "func.func"() <{function_type = () -> (), sym_name = "test_canonicalize_to_attrs"}> ({
// CHECK:             %[[VAL_0:.*]] = "c0_index_dyn"() : () -> index
// CHECK:             %[[VAL_1:.*]] = "c100_index_dyn"() : () -> index
// CHECK:             %[[VAL_2:.*]] = "c2_index_dyn"() : () -> index
// CHECK:             %[[VAL_3:.*]] = "indexing.arange"(%[[VAL_0]]) {operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : index, stopAttr = 100 : index} : (index) -> tensor<?xindex>
// CHECK:             "use1"(%[[VAL_3]]) : (tensor<?xindex>) -> ()
// CHECK:             %[[VAL_4:.*]] = "indexing.arange"(%[[VAL_0]], %[[VAL_1]]) {operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : index} : (index, index) -> tensor<?xindex>
// CHECK:             "use2"(%[[VAL_4]]) : (tensor<?xindex>) -> ()
// CHECK:             %[[VAL_5:.*]] = "indexing.arange"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?xindex>
// CHECK:             "use3"(%[[VAL_5]]) : (tensor<?xindex>) -> ()
// CHECK:             %[[VAL_6:.*]] = "indexing.arange"(%[[VAL_0]]) {operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : index, stopAttr = 100 : index} : (index) -> tensor<?xindex>
// CHECK:             "use4"(%[[VAL_6]]) : (tensor<?xindex>) -> ()
// CHECK:             %[[VAL_7:.*]] = "indexing.arange"(%[[VAL_0]], %[[VAL_1]]) {operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : index} : (index, index) -> tensor<?xindex>
// CHECK:             "use5"(%[[VAL_7]]) : (tensor<?xindex>) -> ()
// CHECK:             "func.return"() : () -> ()
// CHECK:           }) : () -> ()
// CHECK:         }) : () -> ()

module {
  func.func @test_canonicalize_to_attrs() {
    %c100_index = arith.constant 100 : index
    %c2_index = arith.constant 2 : index
    %one = arith.constant 1 : index
    %c0_index_dyn = "c0_index_dyn"() : () -> (index)
    %c100_index_dyn = "c100_index_dyn"() : () -> (index)
    %c2_index_dyn = "c2_index_dyn"() : () -> (index)

    %10 = "indexing.arange"(%c0_index_dyn, %c100_index, %c2_index) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?xindex>
    "use1"(%10) : (tensor<?xindex>) -> ()

    %11 = "indexing.arange"(%c0_index_dyn, %c100_index_dyn, %c2_index) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?xindex>
    "use2"(%11) : (tensor<?xindex>) -> ()

    %12 = "indexing.arange"(%c0_index_dyn, %c100_index_dyn, %c2_index_dyn) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?xindex>
    "use3"(%12) : (tensor<?xindex>) -> ()

    %13 = "indexing.arange"(%c0_index_dyn) {stopAttr = 100 : index, stepAttr = 2 : index, operand_segment_sizes = array<i32: 1, 0, 0>} : (index) -> tensor<?xindex>
    "use4"(%13) : (tensor<?xindex>) -> ()

    %14 = "indexing.arange"(%c0_index_dyn, %c100_index_dyn) {stepAttr = 2 : index, operand_segment_sizes = array<i32: 1, 1, 0>} : (index, index) -> tensor<?xindex>
    "use5"(%14) : (tensor<?xindex>) -> ()

    return
  }
}

// -----

// CHECK-LABEL:   "builtin.module"() ({
// CHECK:           "func.func"() <{function_type = () -> (), sym_name = "test_fold"}> ({
// CHECK:             %[[VAL_0:.*]] = "arith.constant"() <{value = dense<[0, 2, 4, 6, 8, 10, {{.*}}, 98]> : tensor<50xindex>}> : () -> tensor<50xindex>
// CHECK:             "use1"(%[[VAL_0]]) : (tensor<50xindex>) -> ()
// CHECK:             "use2"(%[[VAL_0]]) : (tensor<50xindex>) -> ()
// CHECK:             "use3"(%[[VAL_0]]) : (tensor<50xindex>) -> ()
// CHECK:             "use4"(%[[VAL_0]]) : (tensor<50xindex>) -> ()
// CHECK:             "func.return"() : () -> ()
// CHECK:           }) : () -> ()
// CHECK:         }) : () -> ()


module {
  func.func @test_fold() {
    %c0_index = arith.constant 0 : index
    %c100_index = arith.constant 100 : index
    %c2_index = arith.constant 2 : index

    %4 = "indexing.arange"(%c0_index, %c100_index, %c2_index) {operand_segment_sizes = array<i32: 1, 1, 1>} : (index, index, index) -> tensor<?xindex>
    "use1"(%4) : (tensor<?xindex>) -> ()
    %5 = "indexing.arange"(%c0_index, %c2_index) {stopAttr = 100 : index, operand_segment_sizes = array<i32: 1, 0, 1>} : (index, index) -> tensor<?xindex>
    "use2"(%5) : (tensor<?xindex>) -> ()
    %si = "indexing.arange"(%c0_index) {stopAttr = 100 : index, stepAttr = 2 : index, operand_segment_sizes = array<i32: 1, 0, 0>} : (index) -> tensor<?xindex>
    "use3"(%si) : (tensor<?xindex>) -> ()
    %se = "indexing.arange"() {startAttr = 0 : index, stopAttr = 100 : index, stepAttr = 2 : index, operand_segment_sizes = array<i32: 0, 0, 0>} : () -> tensor<50xindex>
    "use4"(%se) : (tensor<50xindex>) -> ()

    return
  }
}
