import numpy as np

import logging
from iree import runtime as ireert
from iree.compiler import compile_str

import iree.compiler.tools

################################################################################
# Hardcoded strategy with the schedule dialect to drive IREE through a file.
################################################################################

# Create a transformation file for the given IR module.
# Atm, this is the simplest way to transport the IR module to the target through
# IREE.
TENSOR_TRANSFORM_SPEC = """
pdl.pattern @isa_linalg.matmul : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation (%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  %3 = attribute "linalg.matmul"
  apply_native_constraint "isEquivalentToOp"(%2, %3 : !pdl.operation, !pdl.attribute)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %input_matmul = match @isa_linalg.matmul
  // %tiling_result contains the tiled op and the linalg_ext.tile op.
  %tiling_result:2 = tile_to_iree_linalg_ext_tile_op %input_matmul {sizes = [8]}
  %outer_parallel = rewrite_iree_linalg_ext_tile_to_in_parallel %tiling_result#1
  // TODO: careful, print gets mangled into IREE atm and results in a corrupt binary.
  // print {name = "After outer parallel"}

  %0 = match @isa_linalg.matmul
  %tiled_linalg_op, %loops:3 = tile %0 {interchange = [0, 1, 2], sizes = [8, 32, 8]}
  %1 = peel_loop %loops#2
  %2 = scalarize %tiled_linalg_op
  vectorize {vectorize_padding = false}
  // TODO: careful, print gets mangled into IREE atm and results in a corrupt binary.
  // print {name = "After vectorize"}

  // TODO: Ideally we would bufferize here but we can't atm.
  // bufferize
  // Bufferize happens at the IREE level on HAL operations, we cannot just
  // call the linalg_transform.bufferize operation here.
  // Instead it happens automatically at the end of the linalg-transform-interp
  // pass.
  
  rewrite_iree_linalg_ext_in_parallel_to_hal %outer_parallel
  // TODO: careful, print gets mangled into IREE atm and results in a corrupt binary.
  // print {name = "After rewrite to HAL"}
}
"""

# These transforms need to occur after bufferization, which we don't control atm.
# TODO: we may want to add a post-bufferization transform module.
MEMREF_TRANSFORM_SPEC = """
pdl.pattern @isa_linalg.copy : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg.generic"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  %3 = attribute "linalg.copy"
  apply_native_constraint "isEquivalentToOp"(%2, %3 : !pdl.operation, !pdl.attribute)
  rewrite %2 with "iree_linalg_transform.apply"
}

pdl.pattern @isa_linalg_ext.in_parallel : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation "linalg_ext.in_parallel"(%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  %outer_parallel = match @isa_linalg_ext.in_parallel
  rewrite_iree_linalg_ext_in_parallel_to_hal %outer_parallel
  // TODO: careful, print gets mangled into IREE atm and results in a corrupt binary.
  // print {name = "After rewrite to HAL"}

  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4, 5], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4, 5, 6], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4, 5, 6, 7], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_to_llvm {enable_amx = false, enable_arm_neon = false, enable_arm_sve = false, enable_async = false, enable_index_optimizations = false, enable_x86vector = false, reassociate_fp_reductions = false}
}
"""

TRANSFORM_SPEC_FILE_NAME = "/tmp/linalg_transform_spec.mlir"
with open(TRANSFORM_SPEC_FILE_NAME, "w") as f:
  f.write(TENSOR_TRANSFORM_SPEC)

################################################################################
# END Hardcoded strategy with the schedule dialect to drive IREE through a file.
################################################################################

# Compile a module.

DOT_ASM = """
func @dot(%lhs: tensor<128x128xf32>, %rhs: tensor<128x128xf32>) -> tensor<128x128xf32> {
     %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    return %0 : tensor<128x128xf32>
}
"""

# TODO: this is currently broken until IREE + transform dialect emit parallel code again.
if False:
  binary = iree.compiler.tools.compile_str(
      DOT_ASM,
      input_type="mhlo",
      target_backends=["dylib"],
      extra_args=[
          '--iree-codegen-use-linalg-transform-interp',
          '--linalg-transform-file-name=' + TRANSFORM_SPEC_FILE_NAME,
          #'-mlir-print-ir-after-all',
          #'-mlir-print-ir-after-change',
      ])
  print(f'Flatbuffer size = {len(binary)}')

  with open('/tmp/binary.vfmb', "wb") as f:
    f.write(binary)

  # Register the module with a runtime context.
  # Use the CPU interpreter (which has the most implementation done):
  config = ireert.Config("dylib")
  ctx = ireert.SystemContext(config=config)
  vm_module = ireert.VmModule.from_flatbuffer(binary)
  ctx.add_vm_module(vm_module)

  # Invoke the function and print the result.
  lhs = np.full((128, 128), 1, dtype=np.float32)
  rhs = np.full((128, 128), 2, dtype=np.float32)
  dot = ctx.modules.module.dot
  res = dot(lhs, rhs)

  np.testing.assert_allclose(res, np.dot(lhs, rhs))
  print('SUCCESS')