from typing import Tuple

# XXX: Remove paths to `jax*` packages installed from pip by Bazel rules.
import sys
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import jax
from jax import lax
from jax import numpy as jnp
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import transform
import pytest

from jasc import jasc

JASC_MLIR_TAG = "bmm"


def _gen_input(shape: Tuple[int, int], dtype=jnp.float64):
  return jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=dtype)


def test_batch_matmul():
  def batch_matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(lax.batch_matmul, JASC_MLIR_TAG)(a, b)

  def schedule(handle: jasc.OpHandle) -> None:
    bmm = handle.match_tag(JASC_MLIR_TAG).select("linalg.batch_matmul")
    loop = bmm.tile(
        loop=jasc.TileLoopKind.FORALL,
        num_threads=[1],
        mapping=[ir.Attribute.parse("#gpu.block<x>")],
    ).loops[0]
    fill = handle.match_tag(JASC_MLIR_TAG).select("linalg.fill")
    fill.fuse_into(loop)
    with handle.match_ops("func.func").apply_patterns():
      transform.ApplyCanonicalizationPatternsOp()
    handle.match_ops("tensor.empty").replace_with_alloc_tensor()
    handle = handle.one_shot_bufferize(
        bufferize_function_boundaries=True,
        function_boundary_type_conversion="IdentityLayoutMap",
    )
    handle.map_forall_to_blocks(grid_dims=[1, 1, 1], generate_gpu_launch=True)

  a = _gen_input((4, 32, 32))
  b = _gen_input((4, 32, 32))

  jit_batch_matmul = jasc.jit(batch_matmul, schedule)
  chex.assert_gpu_available()
  chex.assert_trees_all_close(
      jit_batch_matmul(a, b), lax.batch_matmul(a, b), rtol=1e-5
  )


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
