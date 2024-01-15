# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Integration test of full schedules for `jax.numpy.matmul`."""

from typing import Tuple

# XXX: Remove paths to `jax*` packages installed from pip by Bazel rules.
import sys
sys.path = [p for p in sys.path if "/pip_deps_jax" not in p]

import chex
import jax
from jax import numpy as jnp
from jaxlib.mlir.dialects import transform
from jaxlib.mlir.dialects.transform import (
    vector,
)
import pytest

from jasc import jasc


def _gen_input(shape: Tuple[int, int], dtype=jnp.float64):
  return jax.random.uniform(jax.random.PRNGKey(0), shape, dtype=dtype)


# ===----------------------------------------------------------------------=== #
# ntv@ matmult schedule.
# ===----------------------------------------------------------------------=== #
# This schedule re-implements a schedule originally written by ntv in C++.
# ===----------------------------------------------------------------------=== #
def test_ntv_matmul():
  def build_matmul_strategy_block_distribution(
      func: jasc.OpHandle, tile_sizes, mapping
  ):
    tiled, loops = func.match_ops("linalg.generic").tile(
        loop=jasc.TileLoopKind.FORALL, tile_sizes=tile_sizes, mapping=mapping
    )
    forall = loops[0]
    build_canonicalize_and_enabling_transforms(func)
    fused_fill = func.match_ops("linalg.fill").fuse_into(forall)
    build_canonicalize_and_enabling_transforms(func)
    return (fused_fill, tiled, forall)

  def build_map_top_level_forall_to_blocks(
      func: jasc.OpHandle, grid_dims
  ) -> jasc.OpHandle:
    build_canonicalize_and_enabling_transforms(func)
    return func.map_forall_to_blocks(
        grid_dims=grid_dims, generate_gpu_launch=True
    )

  def build_bufferize(target: jasc.OpHandle):
    build_canonicalize_and_enabling_transforms(target)
    target.match_ops("tensor.empty").replace_with_alloc_tensor()
    # TODO: We have to bufferize on the module in order to bufferize the
    #       function boundaries but we loose the handle to the function that
    #       way. Find away around that.
    module = target.get_parent_op("builtin.module").one_shot_bufferize(
        bufferize_function_boundaries=True,
        function_boundary_type_conversion="IdentityLayoutMap",
    )
    target = module.match_ops("func.func")  # XXX: get our function back
    build_canonicalize_and_enabling_transforms(target)
    return target

  def build_canonicalize_and_enabling_transforms(target: jasc.OpHandle):
    with target.apply_patterns():
      transform.structured.ApplyTilingCanonicalizationPatternsOp()
      transform.loop.ApplyForLoopCanonicalizationPatternsOp()
      transform.ApplyCanonicalizationPatternsOp()
    target.apply_cse()

  def build_transform_strategy(target: jasc.OpHandle):
    tsb_x = 16
    tsb_y = 8
    num_blocks_x = (789 + tsb_x + 1) // tsb_x
    num_blocks_y = (123 + tsb_y + 1) // tsb_y
    mapping = ["#gpu.block<x>", "#gpu.block<y>"]
    _, matmul, _ = build_matmul_strategy_block_distribution(
        target, [num_blocks_x, num_blocks_y], mapping
    )

    matmul.tile(loop=jasc.TileLoopKind.FOR, tile_sizes=(0, 0, 32))
    build_canonicalize_and_enabling_transforms(target)

    target = build_bufferize(target)

    target = build_map_top_level_forall_to_blocks(
        target, [num_blocks_x, num_blocks_y, 1]
    )
    build_canonicalize_and_enabling_transforms(target)

    return ((num_blocks_x, num_blocks_y, 1), (1, 1, 1))

  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  def schedule(handle: jasc.OpHandle) -> None:
    func = handle.match_ops("func.func")
    build_transform_strategy(func)
    del handle

  a = _gen_input((123, 789))
  b = _gen_input((789, 123))
  jit_matmul = jasc.jit(matmul, schedule)
  chex.assert_gpu_available()
  chex.assert_trees_all_close(jit_matmul(a, b), jnp.matmul(a, b), rtol=1e-5)


# ===----------------------------------------------------------------------=== #
# springerm@ matmul schedule.
# ===----------------------------------------------------------------------=== #
# This schedule re-implements and extends the schedule from upstream at
# https://github.com/llvm/llvm-project/blob/96ff0255f/mlir/test/Dialect/Linalg/matmul-shared-memory-padding.mlir
# ===----------------------------------------------------------------------=== #
def test_springerm_matmul():
  def matmul(a: jax.Array, b: jax.Array) -> jax.Array:
    return jasc.tag(jax.numpy.matmul, "matmul")(a, b)

  def schedule(handle: jasc.OpHandle) -> None:
    # Fuse `linalg.fill` into `linalg.generic` (matmul) and tile across
    # blocks.
    tiled_matmul, loops = handle.match_ops("linalg.generic").tile(
        loop=jasc.TileLoopKind.FORALL,
        tile_sizes=[64, 64],
        mapping=["#gpu.block<y>", "#gpu.block<x>"],
    )
    tiled_fused_fill = handle.match_ops("linalg.fill").fuse_into(loops[0])

    # Tile matmul a second time.
    tiled_matmul, _ = tiled_matmul.tile(
        loop=jasc.TileLoopKind.FOR, tile_sizes=(0, 0, 16)
    )

    # Pad matmul.
    tiled_padded_matmul, pad, copy_back = tiled_matmul.pad(
        padding_values=[0.0, 0.0, 0.0],
        padding_dimensions=[0, 1, 2],
        pack_paddings=[1, 1, 1],
        copy_back_op=jasc.PadCopyBackOp.LINALG_COPY,
    )

    # Tile ops across threads and vectorize.
    tiled_padded_matmul.tile(
        loop=jasc.TileLoopKind.FORALL,
        num_threads=[8, 32],
        mapping=["#gpu.thread<y>", "#gpu.thread<x>"],
    ).tiled_op.vectorize()
    tiled_fused_fill.tile(
        loop=jasc.TileLoopKind.FORALL,
        num_threads=[8, 32],
        mapping=["#gpu.thread<y>", "#gpu.thread<x>"],
    ).tiled_op.vectorize()

    # Map `tensor.pad` and copy back to threads.
    pad_forall_op, tiled_pad_op = pad.map_copy_to_threads(
        total_num_threads=256, desired_bit_alignment=128
    )
    copy_back.map_copy_to_threads(
        total_num_threads=256, desired_bit_alignment=128
    )

    # Vectorize padding ops.
    tiled_pad_op.vectorize(vector_sizes=[128, 4])

    # Assign shared memory buffer to padding.
    padding_bufferization = pad_forall_op.bufferize_to_allocation(
        memory_space=3,
        bufferize_destination_only=True,
        alloc_op="memref.alloca",
    )

    # Transform `memref.alloca`s to `memref.global`s in order to work in the
    # `gpu.launch`.
    foreach = padding_bufferization.new_ops.foreach(
        [transform.AnyOpType.get()]
    )
    with foreach.body as op:
      alloca = op.match_ops("memref.alloca")
      jasc.yield_(alloca)
    foreach.results[0].alloca_to_global()

    # Bufferize the whole function.
    (
        handle.match_ops("func.func")
        .eliminate_empty_tensors()
        .apply_dce()
        .apply_cse()
    )

    handle.match_ops("tensor.empty").replace_with_alloc_tensor()
    bufferized = handle.one_shot_bufferize(
        bufferize_function_boundaries=True,
        function_boundary_type_conversion="IdentityLayoutMap",
    )

    # Apply vectorization to copy back from shared memory.
    # TODO: Find a way to retain the handle to linalg.copy throughout
    #       bufferization.
    func = bufferized.match_ops("func.func")
    func.match_ops("linalg.copy").vectorize(vector_sizes=[128, 4])

    # Canonicalize, cleanup, and vector lowering. This step also removes
    # buffer self-copies.
    with func.apply_patterns(apply_cse=True):
      transform.ApplyCanonicalizationPatternsOp()
      vector.ApplyVectorReductionToContractPatternsOp()
      vector.ApplyLowerMaskedTransfersPatternsOp()
      vector.ApplyTransferPermutationPatternsOp()
      vector.ApplyVectorReductionToContractPatternsOp()

    # Map the `scf.forall`s to GPU blocks and threads.
    func.map_forall_to_blocks(
        grid_dims=[16, 16, 1],
        generate_gpu_launch=True,
    ).map_nested_forall_to_threads(block_dims=[32, 8, 1])

    # Some more clean-ups.
    func = bufferized.match_ops(
        "func.func"
    ).hoist_redundant_vector_transfers()
    with func.apply_patterns(apply_cse=True):
      vector.ApplyTransferToScfPatternsOp(
          max_transfer_rank=1, full_unroll=True
      )

    del handle

  jit_matmul = jasc.jit(matmul, schedule)
  a = _gen_input((1024, 1024), dtype=jnp.float32)
  b = _gen_input((1024, 1024), dtype=jnp.float32)
  chex.assert_gpu_available()
  chex.assert_trees_all_close(jit_matmul(a, b), jnp.matmul(a, b), rtol=1e-5)


if __name__ == "__main__":
    args = sys.argv[1:] or ["-s", "-v"]
    sys.exit(pytest.main([__file__] + args))
