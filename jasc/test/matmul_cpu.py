# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration test of full schedules for `jax.numpy.matmul`."""

from typing import Tuple
import sys

# XXX: Remove paths to `jax*` packages installed from pip by Bazel rules.
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import jax
from jax import numpy as jnp
from jaxlib.mlir.dialects import transform
from jaxlib.mlir.dialects.transform import (
    loop,
    memref,
    structured,
    tensor,
    vector,
)
import pytest

from jasc import jasc


def _gen_input(shape: Tuple[int, int], dtype=jnp.float64):
  return jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=dtype)


# ===----------------------------------------------------------------------=== #
# zinenko@ matmul schedules.
# ===----------------------------------------------------------------------=== #
# These schedules reimplements internal schedules from zinenko@ 
# ===----------------------------------------------------------------------=== #

@pytest.mark.parametrize("m,k,n", [
    (4,  768, 2304),
    (4, 2304,  768),
    (4,  768, 3072),
    (4, 3072,  768),
])
def test_zinenko_matmul_f32(m, k, n):
  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  def schedule(module: jasc.OpHandle) -> None:
    # Tile matmul.
    # Note: Unlike the original schedule, we tile to `scf.forall` such that we
    #       can fuse the `linalg.fill`, which the other schedule doesn't have.
    tiled_matmul, loops = module.match_ops("linalg.generic").tile(
        loop=jasc.TileLoopKind.FORALL, tile_sizes=(0, 16)
    )
    module.match_ops("linalg.fill").fuse_into(loops[0])

    # Tile matmul again, then interchange.
    tiled_matmul.tile(
        loop=jasc.TileLoopKind.FOR,
        tile_sizes=(0, 0, 8),
    ).tiled_op.interchange([0, 2, 1]).vectorize()

    # Manual clean-up.
    func = module.match_ops("func.func")
    with func.apply_patterns():
      transform.ApplyCanonicalizationPatternsOp()
      structured.ApplyTilingCanonicalizationPatternsOp()
    func.apply_cse()
    func.match_ops("LoopLikeInterface").apply_licm()
    with func.apply_patterns():
      structured.ApplyFoldUnitExtentDimsViaReshapesPatternsOp()

    # Vectorize function.
    func.vectorize_children_and_apply_patterns(vectorize_padding=True)

    # Hoist redundant transforms.
    with func.apply_patterns():
      transform.ApplyCanonicalizationPatternsOp()
      tensor.ApplyFoldTensorSubsetOpsIntoVectorTransfersPatternsOp()
    func.apply_cse()
    func.hoist_redundant_vector_transfers()

    # Bufferize.
    module.one_shot_bufferize(
        bufferize_function_boundaries=True,
        function_boundary_type_conversion="IdentityLayoutMap",
    )

    # Turn the `scf.forall` into `scf.for`.
    # Note: The original schedule does not do that since it creates `scf.for`
    #       right away (see above).
    forall = module.match_ops("scf.forall")
    loop.ForallToForOp([transform.AnyOpType.get()], forall.mlir_value)

    # Lowering of vector ops.
    func = module.match_ops("func.func")
    with func.apply_patterns():
      transform.ApplyCanonicalizationPatternsOp()
    with func.apply_patterns():
      vector.ApplyLowerContractionPatternsOp()
      vector.ApplyLowerTransposePatternsOp()
      vector.ApplyLowerTransferPatternsOp()
      vector.ApplyLowerShapeCastPatternsOp()
    with func.apply_patterns():
      vector.ApplyTransferToScfPatternsOp(full_unroll=True)
      memref.ApplyAllocToAllocaOp()

    # Hoist buffers. (Does not have any effect on this input).
    func.buffer_loop_hoisting()

    # Final foldings and clean-up.
    with func.apply_patterns():
      memref.ApplyFoldMemrefAliasOpsPatternsOp()
      transform.ApplyCanonicalizationPatternsOp()
    func.apply_cse()

  jit_matmul = jasc.jit(matmul, schedule)
  a = _gen_input((m, k), dtype=jnp.float32)
  b = _gen_input((k, n), dtype=jnp.float32)
  chex.assert_trees_all_close(jit_matmul(a, b), jnp.matmul(a, b), rtol=1e-5)


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
