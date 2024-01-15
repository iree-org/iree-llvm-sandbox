# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for MLIR diagnostics."""

from __future__ import annotations

from typing import Callable, Sequence

from absl import app
import jax
from jax import numpy as jnp

from jasc import jasc

tests: list[Callable[[], None]] = []


def run(f):
  def test():
    print("\nTEST:", f.__name__)
    f()

  tests.append(test)
  return f


# CHECK-LABEL: TEST: test_location_notes
@run
def test_location_notes():
  def foo(a: jax.Array) -> jax.Array:
    return a

  def schedule(h: jasc.OpHandle) -> None:
    # This is invalid because it applies CSE to the embedded transform script,
    # which can't be modified.
    h.apply_cse()

  # Check that the exception contains notes and locations.
  try:
    data = jnp.full((16, 16), 1.23)
    jasc.lower_to_linalg(foo, data, schedule=schedule)
  except Exception as e:
    print(e)
  # CHECK:      /tmp/mlir_snapshot-{{.*}}.tmp.mlir:
  # CHECK-SAME:   error: cannot apply transform to itself
  # CHECK:      apply_cse to %{{.*}} : !transform.any_op
  # CHECK-NEXT: ^
  # CHECK:      /tmp/mlir_snapshot-{{.*}}.tmp.mlir:
  # CHECK-SAME:   note: target payload op


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  for test_fun in tests:
    test_fun()


if __name__ == "__main__":
  app.run(main)
