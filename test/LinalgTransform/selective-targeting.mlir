// RUN: mlir-proto-opt %s -linalg-interp-transforms | FileCheck %s

// CHECK-LABEL: func @matmul_tensors(
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32>,
  %arg3: tensor<128x128xf32>, %arg4: tensor<128x128xf32>, %arg5: tensor<128x128xf32>,
  %arg6: tensor<128x128xf32> {linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // This operation is marked for tiling only.
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-3: tensor.extract_slice
  // CHECK: linalg.matmul
  // CHECK-SAME: -> tensor<4x4xf32>
  %0 = linalg.matmul { test.attrA} 
                      ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  // This operation is marked for tiling and vectorization.
  // CHECK-COUNT-3: scf.for
  // CHECK-COUNT-3: vector.transfer_read
  // CHECK: vector.contract
  // CHECK-NOT: linalg.matmul
  // CHECK: vector.transfer_write
  %1 = linalg.matmul { test.attrA, test.attrC}
                      ins(%arg3, %arg4: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg5: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  // This operation is marked for vectorization only.
  // CHECK-NOT: scf.for
  // CHECK-COUNT-3: vector.transfer_read
  // CHECK: vector.contract
  // CHECK-SAME: into vector<128x128xf32>
  // CHECK: vector.transfer_write
  %2 = linalg.matmul { test.attrC}
                      ins(%0, %1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg6: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %2 : tensor<128x128xf32>
}

// Match matmul operations inside @matmul_tensors with test.attrA set.
pdl.pattern @pdl_target_attrA : benefit(1) {
  %args = pdl.operands
  %results = pdl.types
  %attr = pdl.attribute
  %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrA" = %attr}-> (%results : !pdl.range<type>)
  pdl.apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  pdl.rewrite %0 with "linalg_transform.apply"
}

// Match matmul operations inside @matmul_tensors with test.attrC set.
pdl.pattern @pdl_target_attrC : benefit(1) {
  %args = pdl.operands
  %results = pdl.types
  %attr = pdl.attribute
  %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) {"test.attrC" = %attr}-> (%results : !pdl.range<type>)
  pdl.apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  pdl.rewrite %0 with "linalg_transform.apply"
}

linalg_transform.sequence {
  tile when @pdl_target_attrA {sizes = [4, 4, 4], pad = false}
  vectorize when @pdl_target_attrC
}
