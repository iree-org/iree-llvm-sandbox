# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, Sequence

from absl import app
from jaxlib.mlir import ir, passmanager
from jaxlib.mlir.dialects import transform

from jaxlib.mlir.dialects import jasc as jd
from jaxlib.mlir.dialects.transform import jasc_transform_ops as jto

tests: list[Callable[[], None]] = []


def run(f):
  def test():
    print("\nTEST:", f.__name__)
    f()

  tests.append(test)
  return f


# CHECK-LABEL: test_register_transform_dialect_extension
@run
def test_register_transform_dialect_extension() -> None:
  with ir.Context() as ctx, ir.Location.unknown():
    jto.register_transform_dialect_extension(ctx)
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      sequence = transform.SequenceOp(
          transform.FailurePropagationMode.Propagate,
          [],
          transform.AnyOpType.get(),
      )
      with ir.InsertionPoint(sequence.body):
        jto.MatchTagOp(sequence.bodyTarget, ["tag"])
        transform.YieldOp([])
    module.operation.verify()
    print(module)
    # CHECK: transform.sequence
    # CHECK:   transform.jasc.match_tag


# CHECK-LABEL: test_register_and_load_dialect
@run
def test_register_and_load_dialect() -> None:
  with ir.Context(), ir.Location.unknown():
    jd.register_and_load_dialect()
    module = ir.Module.create()
    with ir.InsertionPoint(module.body):
      op = jd.TagRegionOp([], "tag")
      op.body.blocks.append()
    with ir.InsertionPoint(op.body.blocks[0]):
      jd.ReturnOp([])
    module.operation.verify()
    print(module)
    # CHECK:      jasc.tag_region "tag" {
    # CHECK-NEXT: }


# CHECK-LABEL: test_register_lowering_passes
@run
def test_register_lowering_passes() -> None:
  with ir.Context(), ir.Location.unknown():
    jd.register_lowering_passes()
    module = ir.Module.create()
    pm = passmanager.PassManager.parse(
        "builtin.module(jasc-remove-copy-to-out-params)"
    )
    pm.run(module.operation)
    print(module)
    # CHECK: module


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  for test_fun in tests:
    test_fun()


if __name__ == "__main__":
  app.run(main)
