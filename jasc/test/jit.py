# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for JASC jit."""
from __future__ import annotations

import sys

# Remove paths to `jax*` packages installed from pip. See requirements.txt.
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import jax
import pytest

from jasc import jasc

jasc.set_auto_normalization(False)


def _gen_input(shape: tuple[int, int], dtype=jax.numpy.float64):
  return jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=dtype)


def test_jit_matmul_jax_func() -> None:
  """Jasc jit of a jax function without an additional schedule."""

  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  jit_matmul = jasc.jit(matmul)
  a = _gen_input((64, 64))
  b = _gen_input((64, 64))
  chex.assert_trees_all_close(
      jit_matmul(a, b), jax.numpy.matmul(a, b), rtol=1e-5
  )

def test_jit_matmul_jax_func_schedule() -> None:
  """Jasc jit of a jax function with a simple schedule."""

  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  def schedule(handle: jasc.OpHandle) -> None:
    handle.match_ops("linalg.generic").tile(
        loop=jasc.TileLoopKind.FOR, tile_sizes=[32]
    )

  jit_matmul = jasc.jit(matmul, schedule)
  a = _gen_input((64, 64))
  b = _gen_input((64, 64))
  chex.assert_trees_all_close(
      jit_matmul(a, b), jax.numpy.matmul(a, b), rtol=1e-5
  )


def test_jit_matmul_mlir() -> None:
  """Jasc jit of an mlir module that stems from a jax function."""

  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  a = _gen_input((64, 64))
  b = _gen_input((64, 64))

  module = jasc.lower_to_linalg(matmul, a, b)
  jit_matmul = jasc.jit(matmul, module=module)

  chex.assert_trees_all_close(
      jit_matmul(a, b), jax.numpy.matmul(a, b), rtol=1e-5
  )


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
