// RUN: mlir-proto-opt %s -test-staged-pattern-rewrite-driver="test-case=1" -allow-unregistered-dialect | FileCheck %s --check-prefix=CASE1
// RUN: mlir-proto-opt %s -test-staged-pattern-rewrite-driver="test-case=2" -allow-unregistered-dialect | FileCheck %s --check-prefix=CASE2
// RUN: mlir-proto-opt %s -test-staged-pattern-rewrite-driver="test-case=3" -allow-unregistered-dialect | FileCheck %s --check-prefix=CASE3

// CASE1-LABEL: func @test1() {
func @test1() {
  // CASE1: "start"() {__test_attr__ = 0 : i32} : () -> ()
  "start"() : () -> ()
}

// CASE2-LABEL: func @test2() {
func @test2() {
  // CASE2: "start"() {__test_attr__ = 2 : i32} : () -> ()
  "start"() : () -> ()
}

// CASE3-LABEL: func @test3() {
func @test3() {
  // CASE3: "start"() {__test_attr__ = 1 : i32} : () -> ()
  // CASE3: "step_1"() : () -> ()
  "start"() : () -> ()
  "start"() : () -> ()
}
