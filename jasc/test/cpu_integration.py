"""Jasc tests common to all platforms."""

from collections.abc import Mapping
import sys

# XXX: Remove paths to `jax*` packages installed from pip by Bazel rules.
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import jax
from jax import numpy as jnp
import pytest

from jasc import jasc


def _unit_schedule(handle: jasc.OpHandle) -> None:
  """A schedule that does nothing."""
  del handle


def test_jit_pass():
  def foo() -> None:
    pass

  assert jasc.jit(foo, _unit_schedule)() is None


def test_jit_single_input_output():
  def foo(x: jax.Array) -> jax.Array:
    return x

  x = jnp.array([1, 2, 3])
  chex.assert_trees_all_equal(jasc.jit(foo, _unit_schedule)(x), x)


def test_jit_single_op():
  def foo(x: jax.Array) -> jax.Array:
    return x + 1

  chex.assert_trees_all_equal(
      jasc.jit(foo, _unit_schedule)(jnp.array([1, 2, 3])),
      jnp.array([2, 3, 4]),
  )


def test_jit_multiple_inputs():
  def foo(x: jax.Array, y: jax.Array) -> jax.Array:
    return x + y

  x = jnp.array([1, 2, 3])
  y = jnp.array([4, 5, 6])
  chex.assert_trees_all_equal(
      jasc.jit(foo, _unit_schedule)(x, y), jnp.array([5, 7, 9])
  )


def test_jit_multiple_outputs():
  def foo(x: jax.Array) -> tuple[jax.Array, jax.Array]:
    return x + 1, x + 2

  chex.assert_trees_all_equal(
      jasc.jit(foo, _unit_schedule)(jnp.array([1, 2, 3])),
      (jnp.array([2, 3, 4]), jnp.array([3, 4, 5])),
  )


def test_jit_dict_input():
  def foo(x: Mapping[str, jax.Array]) -> jax.Array:
    return x["a"] + x["b"]

  x = {"a": jnp.array([1, 2, 3]), "b": jnp.array([4, 5, 6])}
  chex.assert_trees_all_equal(
      jasc.jit(foo, _unit_schedule)(x), jnp.array([5, 7, 9])
  )


def test_jit_dict_output():
  def foo(x: jax.Array) -> Mapping[str, jax.Array]:
    return {"a": x + 1, "b": x + 2}

  chex.assert_trees_all_equal(
      jasc.jit(foo, _unit_schedule)(jnp.array([1, 2, 3])),
      {"a": jnp.array([2, 3, 4]), "b": jnp.array([3, 4, 5])},
  )


def test_tag_jit():
  def foo(x: jax.Array) -> jax.Array:
    return x + 1

  jit_foo = jasc.jit(jasc.tag(foo, "foo_tag"), _unit_schedule)
  chex.assert_trees_all_equal(
      jit_foo(jnp.array([1, 2, 3])),
      jnp.array([2, 3, 4]),
  )


def test_tag_jax_jit():
  def foo(x: jax.Array) -> jax.Array:
    return x + 1

  jit_foo = jax.jit(jasc.tag(foo, "foo_tag"))
  chex.assert_trees_all_equal(
      jit_foo(jnp.array([1, 2, 3])), jnp.array([2, 3, 4])
  )


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
