// Test error messages of constraints of CreateTabularViewOp.
// RUN: mlir-proto-opt --verify-diagnostics --split-input-file %s

func.func @testNumberOfTypesMismatch() {
  %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  // expected-error@+1 {{'tabular.create_tabular_view' op type mismatch: should return a tabular view with the same number of columns as the number of input memrefs (expected: 1, found: 2).}}
  %view = "tabular.create_tabular_view"(%m1)
    : (memref<3xi32>) -> !tabular.tabular_view<i32,i64>
  return
}

// -----

func.func @testTypeMismatch() {
  %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  // expected-error@+1 {{'tabular.create_tabular_view' op type mismatch: returned tabular view has column type 'i64' at index 0 but should have type 'i32', the element type of the memref at the same index.}}
  %view = "tabular.create_tabular_view"(%m1)
    : (memref<3xi32>) -> !tabular.tabular_view<i64>
  return
}

// -----

func.func @testSizeMismatch() {
  %t1 = arith.constant dense<[0, 1, 2]> : tensor<3xi32>
  %t2 = arith.constant dense<[0, 1]> : tensor<2xi32>
  %m1 = bufferization.to_memref %t1 : memref<3xi32>
  %m2 = bufferization.to_memref %t2 : memref<2xi32>
  // expected-error@+1 {{'tabular.create_tabular_view' op type mismatch: input memrefs cannot have different static shapes (sizes found for dimension 0: 3, 2).}}
  %view = "tabular.create_tabular_view"(%m1, %m2)
    : (memref<3xi32>, memref<2xi32>) -> !tabular.tabular_view<i32, i32>
  return
}
