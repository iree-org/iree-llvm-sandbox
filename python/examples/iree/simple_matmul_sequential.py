# This file assumes that IREE has been built with python extensions.
# Assuming the IREE build directory is ${IREE_BUILD_DIR}, one would export the
# PYTHONPATH command in ${IREE_BUILD_DIR}/.env
#
# We reproduce it here:
# ```
#    PYTHONPATH=${PYTHONPATH}:${IREE_BUILD_DIR}/compiler-api/python_package:${IREE_BUILD_DIR}/bindings/python \
#    python simple_matmul.py
# ```

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
pdl.pattern @isa_matmul : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation (%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)
  %3 = attribute "linalg.matmul"
  apply_native_constraint "isEquivalentToOp"(%2, %3 : !pdl.operation, !pdl.attribute)
  rewrite %2 with "iree_linalg_transform.apply"
}

pdl.pattern @isa_dynamic_matmul : benefit(1) {
  %0 = operands
  %1 = types
  %2 = operation (%0 : !pdl.range<value>)  -> (%1 : !pdl.range<type>)

  %3 = attribute "linalg.matmul"
  apply_native_constraint "isEquivalentToOp"(%2, %3 : !pdl.operation, !pdl.attribute)

  // TODO: can't seem to get the attr to work here.
  //%two = attribute 2 : i64
  //apply_native_constraint "isDimDynamic"(%2, %two : !pdl.operation, !pdl.attribute)

  rewrite %2 with "iree_linalg_transform.apply"
}

iree_linalg_transform.sequence {
  iree_set_num_workgroups_to_one

  %0 = match @isa_matmul
  // TODO: fuse...
  %tiled_linalg_op, %loops:3 = tile %0 {interchange = [0, 1, 2], sizes = [8, 32, 8]}
  %l0 = peel_loop %loops#0
  %l1 = peel_loop %loops#1

  // TODO: scalarize.
  // %1 = match @isa_dynamic_matmul
  // %2 = scalarize %1
  // print { name = "IR after scalarize" } // note: debugging purposes only otherwise the IR gets corrupted and binary fails to build
  vectorize {vectorize_padding = false}

  iree_bufferize

  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4, 5], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4, 5, 6], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  lower_vectors {contraction_lowering = "outerproduct", multireduction_lowering = "innerparallel", split_transfers = "linalg-copy", stages = [1, 2, 3, 4, 5, 6, 7], transpose_avx2_lowering = false, transpose_lowering = "eltwise", unroll_vector_transfers = true}
  
  // IREE does not want to use this.
  // lower_to_llvm {enable_amx = false, enable_arm_neon = false, enable_arm_sve = false, enable_async = false, enable_index_optimizations = false, enable_x86vector = false, reassociate_fp_reductions = false}
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
func @dot(%lhs: tensor<127x128xf32>, %rhs: tensor<128x129xf32>) -> tensor<127x129xf32> {
     %0 = "mhlo.dot"(%lhs, %rhs) : (tensor<127x128xf32>, tensor<128x129xf32>) -> tensor<127x129xf32>
    return %0 : tensor<127x129xf32>
}
"""

binary = iree.compiler.tools.compile_str(
    DOT_ASM,
    input_type="mhlo",
    target_backends=["dylib"],
    extra_args=[
        '--iree-codegen-use-linalg-transform-interp',
        '--linalg-transform-file-name=' + TRANSFORM_SPEC_FILE_NAME,
        # '-print-ir-after-all',
        # '-print-ir-after-change',
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
lhs = np.full((127, 128), 1, dtype=np.float32)
rhs = np.full((128, 129), 2, dtype=np.float32)
dot = ctx.modules.module.dot
res = dot(lhs, rhs)

np.testing.assert_allclose(res, np.dot(lhs, rhs))
print('SUCCESS')
