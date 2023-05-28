// RUN: structured-opt -canonicalize -allow-unregistered-dialect -mlir-print-op-generic -split-input-file %s | FileCheck %s

// CHECK: "builtin.module"() ({
// CHECK:   "func.func"() <{function_type = () -> (), sym_name = "test_canonicalize_to_attrs"}> ({
// CHECK:     %[[C0:.*]] = "c0_i64_dyn"() : () -> i64
// CHECK:     %[[C100:.*]] = "c100_i64_dyn"() : () -> i64
// CHECK:     %[[C2:.*]] = "c2_i64_dyn"() : () -> i64
// CHECK:     %[[ARA:.*]] = "indexing.arange"(%[[C0]]) {operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : i64, stopAttr = 100 : i64} : (i64) -> tensor<?xi64>
// CHECK:     "use1"(%[[ARA]]) : (tensor<?xi64>) -> ()
// CHECK:     %[[ARA:.*]] = "indexing.arange"(%[[C0]], %[[C100]]) {operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : i64} : (i64, i64) -> tensor<?xi64>
// CHECK:     "use2"(%[[ARA]]) : (tensor<?xi64>) -> ()
// CHECK:     %[[ARA:.*]] = "indexing.arange"(%[[C0]], %[[C100]], %[[C2]]) {operand_segment_sizes = array<i32: 1, 1, 1>} : (i64, i64, i64) -> tensor<?xindex>
// CHECK:     "use3"(%[[ARA]]) : (tensor<?xindex>) -> ()
// CHECK:     %[[ARA:.*]] = "indexing.arange"(%[[C0]]) {operand_segment_sizes = array<i32: 1, 0, 0>, stepAttr = 2 : i64, stopAttr = 100 : i64} : (i64) -> tensor<?xindex>
// CHECK:     "use4"(%[[ARA]]) : (tensor<?xindex>) -> ()
// CHECK:     %[[ARA:.*]] = "indexing.arange"(%[[C0]], %[[C100]]) {operand_segment_sizes = array<i32: 1, 1, 0>, stepAttr = 2 : i64} : (i64, i64) -> tensor<?xindex>
// CHECK:     "use5"(%[[ARA]]) : (tensor<?xindex>) -> ()
// CHECK:     "func.return"() : () -> ()
// CHECK:   }) : () -> ()
// CHECK: }) : () -> ()

module {
  func.func @test_canonicalize_to_attrs() {
    %c100_i64 = arith.constant 100 : i64
    %c2_i64 = arith.constant 2 : i64
    %one = arith.constant 1 : i64
    %c0_i64_dyn = "c0_i64_dyn"() : () -> (i64)
    %c100_i64_dyn = "c100_i64_dyn"() : () -> (i64)
    %c2_i64_dyn = "c2_i64_dyn"() : () -> (i64)

    %10 = "indexing.arange"(%c0_i64_dyn, %c100_i64, %c2_i64) {operand_segment_sizes = array<i32: 1, 1, 1>} : (i64, i64, i64) -> tensor<?xindex>
    "use1"(%10) : (tensor<?xindex>) -> ()

    %11 = "indexing.arange"(%c0_i64_dyn, %c100_i64_dyn, %c2_i64) {operand_segment_sizes = array<i32: 1, 1, 1>} : (i64, i64, i64) -> tensor<?xindex>
    "use2"(%11) : (tensor<?xindex>) -> ()

    %12 = "indexing.arange"(%c0_i64_dyn, %c100_i64_dyn, %c2_i64_dyn) {operand_segment_sizes = array<i32: 1, 1, 1>} : (i64, i64, i64) -> tensor<?xindex>
    "use3"(%12) : (tensor<?xindex>) -> ()

    %13 = "indexing.arange"(%c0_i64_dyn) {stopAttr = 100 : i64, stepAttr = 2 : i64, operand_segment_sizes = array<i32: 1, 0, 0>} : (i64) -> tensor<?xindex>
    "use4"(%13) : (tensor<?xindex>) -> ()

    %14 = "indexing.arange"(%c0_i64_dyn, %c100_i64_dyn) {stepAttr = 2 : i64, operand_segment_sizes = array<i32: 1, 1, 0>} : (i64, i64) -> tensor<?xindex>
    "use5"(%14) : (tensor<?xindex>) -> ()

    return
  }
}

// -----

// CHECK: "builtin.module"() ({
// CHECK:   "func.func"() <{function_type = () -> (), sym_name = "test_fold"}> ({
// CHECK:     %[[ARA:.*]] = "arith.constant"() <{value = dense<[0, 2, {{.*}}, 98]> : tensor<50xi64>}> : () -> tensor<50xi64>
// CHECK:     "use1"(%[[ARA]]) : (tensor<50xi64>) -> ()
// CHECK:     "use2"(%[[ARA]]) : (tensor<50xi64>) -> ()
// CHECK:     "use3"(%[[ARA]]) : (tensor<50xi64>) -> ()
//           this doesn't get folded because the return type is index
// CHECK:     %[[ARA:.*]] = "indexing.arange"() {operand_segment_sizes = array<i32: 0, 0, 0>, startAttr = 0 : i64, stepAttr = 2 : i64, stopAttr = 100 : i64} : () -> tensor<50xindex>
// CHECK:     "use4"(%[[ARA]]) : (tensor<50xindex>) -> ()
// CHECK:     "func.return"() : () -> ()
// CHECK:   }) : () -> ()
// CHECK: }) : () -> ()

module {
  func.func @test_fold() {
    %c0_i64 = arith.constant 0 : i64
    %c100_i64 = arith.constant 100 : i64
    %c2_i64 = arith.constant 2 : i64

    %4 = "indexing.arange"(%c0_i64, %c100_i64, %c2_i64) {operand_segment_sizes = array<i32: 1, 1, 1>} : (i64, i64, i64) -> tensor<?xindex>
    "use1"(%4) : (tensor<?xindex>) -> ()
    %5 = "indexing.arange"(%c0_i64, %c2_i64) {stopAttr = 100 : i64, operand_segment_sizes = array<i32: 1, 0, 1>} : (i64, i64) -> tensor<?xindex>
    "use2"(%5) : (tensor<?xindex>) -> ()
    %si = "indexing.arange"(%c0_i64) {stopAttr = 100 : i64, stepAttr = 2 : i64, operand_segment_sizes = array<i32: 1, 0, 0>} : (i64) -> tensor<?xindex>
    "use3"(%si) : (tensor<?xindex>) -> ()
    %se = "indexing.arange"() {startAttr = 0 : i64, stopAttr = 100 : i64, stepAttr = 2 : i64, operand_segment_sizes = array<i32: 0, 0, 0>} : () -> tensor<50xindex>
    "use4"(%se) : (tensor<50xindex>) -> ()

    return
  }
}
