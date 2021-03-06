import numpy as np

import logging
from iree import runtime as ireert
from iree.compiler import compile_str
from iree.compiler.transforms import ireec

import iree.compiler.tools
import iree.compiler.dialects.transform as transform
import iree.compiler.dialects.pdl as pdl
import iree.compiler.ir as ir

# This example is identical to `simple_matmul_sequential.py`, but uses MLIR
# Python bindings to create ops.

module_str = None
with ir.Context() as ctx, ir.Location.unknown(ctx):
  ireec.register_all_dialects(ctx)

  module = ir.Module.create()
  with ir.InsertionPoint(module.body):
    root = transform.WithPDLPatternsOp(root=None)
    with ir.InsertionPoint(root.body.blocks[0]):
      isa_matmul = pdl.PatternOp(benefit = 1, name = "isa_matmul")
      with ir.InsertionPoint(isa_matmul.body):
        args = pdl.OperandsOp()
        types = pdl.TypesOp()
        pdl_op = pdl.OperationOp(args=[args], types=[types])
        op_name = pdl.AttributeOp(value=ir.StringAttr.get("linalg.matmul"))
        pdl.ApplyNativeConstraintOp("isEquivalentToOp", args=[pdl_op, op_name])
        pdl.RewriteOp(pdl_op, "transform.dialect")

      sequence = transform.CanonicalizedSequenceOp(root.body.blocks[0].arguments[0])
      sequence_block = sequence.body.blocks[0]
      with ir.InsertionPoint(sequence_block):
        ir.Operation.create(name="transform.iree.set_num_workgroups_to_one")
        target_match = transform.PDLMatchOp(sequence_block.arguments[0], "isa_matmul")
        # TODO: fuse...
        tiled = transform.TileOp(target=target_match, sizes=[8, 32, 8])
        transform.PeelLoopOp(tiled.results[1])
        transform.PeelLoopOp(tiled.results[2])
        # TODO: Match dynamic matmul and scalarize.
        transform.VectorizeOp(vectorize_padding=False)
        ir.Operation.create(name="transform.iree.bufferize")

        stages = []
        for i in range(1, 8):
          stages.append(i)
          transform.LowerVectorsOp(contraction_lowering="outerproduct",
                                  multireduction_lowering="innerparallel",
                                  split_transfers="linalg-copy",
                                  stages=stages,
                                  transpose_avx2_lowering=False,
                                  transpose_lowering="eltwise",
                                  unroll_vector_transfers=True)

        transform.YieldOp([])
  module_str = str(module)

# ################################################################################
# # Hardcoded strategy with the schedule dialect to drive IREE through a file.
# ################################################################################

TRANSFORM_SPEC_FILE_NAME = "/tmp/linalg_transform_spec.mlir"
with open(TRANSFORM_SPEC_FILE_NAME, "w") as f:
  f.write(module_str)

################################################################################
# END Hardcoded strategy with the schedule dialect to drive IREE through a file.
################################################################################

# Compile a module.

DOT_ASM = """
func.func.func @dot(%lhs: tensor<127x128xf32>, %rhs: tensor<128x129xf32>) -> tensor<127x129xf32> {
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
lhs = np.full((127, 128), 1, dtype=np.float32)
rhs = np.full((128, 129), 2, dtype=np.float32)
dot = ctx.modules.module.dot
res = dot(lhs, rhs)

np.testing.assert_allclose(res, np.dot(lhs, rhs))
print('SUCCESS')
