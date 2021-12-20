// RUN: mlir-proto-opt -linalg-interp-transforms %s | FileCheck %s

// CHECK-LABEL: func @matmul_tensors(
// CHECK-SAME:    %[[TA:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TB:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:    %[[TC:[0-9a-z]+]]: tensor<128x128xf32>
// CHECK-SAME:  -> tensor<128x128xf32> {
func @matmul_tensors(
  %arg0: tensor<128x128xf32>, %arg1: tensor<128x128xf32>, %arg2: tensor<128x128xf32> { linalg.inplaceable = true})
    -> tensor<128x128xf32> {
  // CHECK: %[[VA:.*]] = vector.transfer_read %[[TA]]
  // CHECK: %[[VB:.*]] = vector.transfer_read %[[TB]]
  // CHECK: %[[VC:.*]] = vector.transfer_read %[[TC]]
  // CHECK: %[[VCU:.*]] = vector.contract {{.*}} %[[VA]], %[[VB]], %[[VC]]
  // CHECK: vector.transfer_write %[[VCU]], %[[TC]]
  %0 = linalg.matmul  ins(%arg0, %arg1: tensor<128x128xf32>, tensor<128x128xf32>)
                     outs(%arg2: tensor<128x128xf32>)
    -> tensor<128x128xf32>

  return %0 : tensor<128x128xf32>
}


pdl.pattern @pdl_target : benefit(1) {
  %args = pdl.operands
  %results = pdl.types
  %0 = pdl.operation "linalg.matmul"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
  pdl.apply_native_constraint "nestedInFunc"[@matmul_tensors](%0 : !pdl.operation)
  // TODO: we don't want this, but it is the required terminator for pdl.pattern
  pdl.rewrite %0 with "linalg_transform.apply"
}

linalg_transform.apply {
^bb0(%arg0: !pdl.operation):
  linalg_transform.sequence {
    %1 = vectorize %arg0 {vectorize_padding = true}
    //bufferize
    //lower_vectors { multireduction_lowering = "innerreduce"}
    //lower_to_llvm
  }
} when {
  linalg_transform.pdl_match @pdl_target
}
