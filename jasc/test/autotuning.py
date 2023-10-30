"""Tests for the JASC autotuning utilities."""

from typing import Tuple
import sys

# XXX: Remove paths to `jax*` packages installed from pip by Bazel rules.
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import jax
from jax import numpy as jnp
import pytest

from jaxlib.mlir import ir
from jaxlib.mlir.dialects import linalg

from jasc import jasc
from jasc import tuner


def _gen_input(shape: Tuple[int, int], dtype=jnp.float64):
  return jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=dtype)


def test_matmul_1D_tiling_tuning() -> None:
  """Tests the tuning of the tile size in a 64x64 matmul."""

  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  def schedule(module: jasc.OpHandle) -> None:
    matmul = module.match_ops(linalg.GenericOp)
    tile_size = jasc.tuning_param(1)
    with jasc.autonormalize(False):
      matmul.tile(loop=jasc.TileLoopKind.FOR, tile_sizes=[tile_size])

  with ir.Context():
    a = _gen_input((64, 64), dtype=jnp.float32)
    b = _gen_input((64, 64), dtype=jnp.float32)

    tuna = tuner.FooTuner(matmul, schedule, inputs=[a, b], budget=10)
    best_time, tuned_fun, best_schedule, times = tuna.tune()

    chex.assert_trees_all_close(
        jax.jit(matmul)(a, b), tuned_fun(a, b), rtol=1e-5
    )
    print(f"times: \n{times}")
    print(f"best time: {best_time}")


def testmatmul_2D_tiling_tuning() -> None:
  """Tests the tuning of the tile sizes in a 64x64 matmul."""

  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  def schedule(module: jasc.OpHandle) -> None:
    matmul = module.match_ops(linalg.GenericOp)
    tile_size = jasc.tuning_param(1)
    with jasc.autonormalize(False):
      matmul.tile(loop=jasc.TileLoopKind.FOR, tile_sizes=[tile_size, tile_size])

  with ir.Context():
    a = _gen_input((64, 64), dtype=jnp.float32)
    b = _gen_input((64, 64), dtype=jnp.float32)

    tuna = tuner.FooTuner(matmul, schedule, inputs=[a, b], budget=10)
    best_time, tuned_fun, best_schedule, times = tuna.tune()

    chex.assert_trees_all_close(
        jax.jit(matmul)(a, b), tuned_fun(a, b), rtol=1e-5
    )

    print(f"times: \n{times}")
    print(f"best time: {best_time}")


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
