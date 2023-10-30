// RUN: jasc-opt %s -jasc-apply-transform-script \
// RUN:   | FileCheck %s


// Test ported from: 
// third_party/iree/compiler/src/iree/compiler/Codegen/Common/test/transform_dialect_apply_pattern_op.mlir
// CHECK-LABEL: @pad_fill_to_fill
func.func @pad_fill_to_fill(%arg0: tensor<31x62xf32>) -> tensor<32x64xf32> {
  // Check that a pad of a fill with the same constant is replaced by a
  // bigger fill.
  // CHECK-DAG: %[[FILL_CST:.*]] = arith.constant 0.0{{0*e\+00}} : f32
  // CHECK-DAG: %[[EMPTY:.*]] = tensor.empty() : tensor<32x64xf32>
  // CHECK: %[[PADDED_FILL:.*]] = linalg.fill ins(%[[FILL_CST]] : f32) outs(%[[EMPTY]] : tensor<32x64xf32>) -> tensor<32x64xf32>
  // CHECK: return %[[PADDED_FILL]]
  %cst = arith.constant 0.0 : f32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %fill = linalg.fill ins(%cst : f32) outs(%arg0 : tensor<31x62xf32>) -> tensor<31x62xf32>
  %padded = tensor.pad %fill low[%c0, %c0] high[%c1, %c2] {
    ^bb0(%arg3: index, %arg4: index):
      tensor.yield %cst : f32
  } : tensor<31x62xf32> to tensor<32x64xf32>
  return %padded : tensor<32x64xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.apply_patterns to %0 {
    transform.apply_patterns.fold_fill_into_pad
    transform.apply_patterns.linalg.tiling_canonicalization
    transform.apply_patterns.scf.for_loop_canonicalization
  } : !transform.any_op
}
