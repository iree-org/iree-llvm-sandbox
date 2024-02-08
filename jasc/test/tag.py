# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Jasc tag primitive based schedule tests."""
from __future__ import annotations
from typing import Callable, Sequence

# Remove paths to `jax*` packages installed from pip. See requirements.txt.
import sys
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

from absl import app
import chex
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


# CHECK-LABEL: TEST: test_match_tag
@run
def test_match_tag():
  """Tests that a schedule based on matching jasc.tag primitives is applied correctly."""

  def foo(a: jax.Array) -> jax.Array:
    b = jax.lax.abs(jasc.tag(lambda x: x + 1, "b")(a))
    c = jasc.tag(lambda x: x * x, "c")(b)
    return c

  input_0 = jnp.full((16, 16), 1.23)

  def schedule(h: jasc.OpHandle) -> None:
    # Disable autonormalization here as it interferes with tags attached to
    # constants.
    with jasc.autonormalize(False):
      h.match_tag("b").tile(loop=jasc.TileLoopKind.FOR, tile_sizes=(2, 4))
      h.match_tag("c").tile(loop=jasc.TileLoopKind.FOR, tile_sizes=(4, 2))

  # Check for correct schedule application
  foo_linalg = jasc.lower_to_linalg(foo, input_0, schedule=schedule)

  # CHECK-LABEL:   func.func public @main(
  # CHECK-SAME:                           %[[VAL_0:.*]]:
  # CHECK:           %[[VAL_1:.*]] = arith.constant {jasc_tags = ["b"]}
  # CHECK:           %[[VAL_2:.*]] = scf.for %[[VAL_3:.*]] = {{.*}} to
  # CHECK:             %[[VAL_4:.*]] = scf.for %[[VAL_5:.*]] = {{.*}} to
  # CHECK-SAME:                                 iter_args(%[[VAL_6:.*]] =
  # CHECK:               %[[VAL_7:.*]] = tensor.extract_slice %[[VAL_0]]{{\[}}%[[VAL_3]], %[[VAL_5]]] [2, 4] [1, 1]
  # CHECK:               %[[VAL_8:.*]] = tensor.extract_slice %[[VAL_1]]{{\[}}%[[VAL_3]], %[[VAL_5]]] [2, 4] [1, 1]
  # CHECK:               %[[VAL_9:.*]] = tensor.extract_slice %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_5]]] [2, 4] [1, 1]
  # CHECK:               %[[VAL_10:.*]] = linalg.map { arith.addf {jasc_tags = ["b"]} } ins(%[[VAL_7]], %[[VAL_8]] :
  # CHECK-SAME:                                                        outs(%[[VAL_9]]
  # CHECK:               %[[VAL_11:.*]] = tensor.insert_slice %[[VAL_10]] into %[[VAL_6]]{{\[}}%[[VAL_3]], %[[VAL_5]]] [2, 4] [1, 1]
  # CHECK:               scf.yield %[[VAL_11]]
  # CHECK:             }
  # CHECK:             scf.yield %[[VAL_4]]
  # CHECK:           }
  # CHECK:           %[[VAL_12:.*]] = linalg.map { math.absf } ins(%[[VAL_2]]
  # CHECK:           %[[VAL_13:.*]] = scf.for %[[VAL_14:.*]] = %{{.*}} to
  # CHECK:             %[[VAL_15:.*]] = arith.constant 16 : index
  # CHECK:             %[[VAL_16:.*]] = scf.for %[[VAL_17:.*]] = %{{.*}} to
  # CHECK-SAME:                                 iter_args(%[[VAL_18:.*]] =
  # CHECK:               %[[VAL_19:.*]] = tensor.extract_slice %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_17]]] [4, 2] [1, 1]
  # CHECK:               %[[VAL_20:.*]] = tensor.extract_slice %[[VAL_12]]{{\[}}%[[VAL_14]], %[[VAL_17]]] [4, 2] [1, 1]
  # CHECK:               %[[VAL_21:.*]] = tensor.extract_slice %[[VAL_18]]{{\[}}%[[VAL_14]], %[[VAL_17]]] [4, 2] [1, 1]
  # CHECK:               %[[VAL_22:.*]] = linalg.map { arith.mulf {jasc_tags = ["c"]} } ins(%[[VAL_19]], %[[VAL_20]]
  # CHECK-SAME:                                                       outs(%[[VAL_21]]
  # CHECK:               %[[VAL_23:.*]] = tensor.insert_slice %[[VAL_22]] into %[[VAL_18]]{{\[}}%[[VAL_14]], %[[VAL_17]]] [4, 2] [1, 1]
  # CHECK:               scf.yield %[[VAL_23]]
  # CHECK:             }
  # CHECK:             scf.yield %[[VAL_16]]
  # CHECK:           }
  # CHECK:           return %[[VAL_13]]
  # CHECK:         }
  print(foo_linalg)

  # Test for similar results of jasc and jax jit
  jax_res = jax.jit(foo)(input_0)
  jasc_res = jasc.jit(foo, schedule)(input_0)
  chex.assert_trees_all_equal(jasc_res, jax_res)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  for test_fun in tests:
    test_fun()


if __name__ == "__main__":
  app.run(main)
