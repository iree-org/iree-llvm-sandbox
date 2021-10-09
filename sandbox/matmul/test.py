# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.transforms import *
from .util import *


class TestExpert(Expert):

  def __init__(self, tiling_transforms):
    self.tiling_transforms = tiling_transforms

  def transforms(self) -> List[Transform]:
    return self.tiling_transforms + [Bufferize(), LowerVectors(), LowerToLLVM()]


# TODO: Check generate code for basic code quality, e.g., no linalg.copy.

# No tiling.
expert_no_tiling = TestExpert([])

# 1 level of tiling.
expert_tile_1 = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 24],
        pad=[],
        peel=False),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 1 level of tile and interchange.
expert_tile_and_interchange_1 = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 24],
        tile_interchange=[2, 0, 1],
        pad=[],
        peel=False),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 1 level of tiling, peel, scalarize the remaining dynamic dims.
expert_tile_1_peel_scalarize = TestExpert([
    Tile(
        'matmul_on_tensors', 'linalg.matmul', tile_sizes=[8], pad=[], peel=[0]),
    Tile('matmul_on_tensors', 'linalg.matmul', scalarize_dyn_dims=True),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 1 level of tiling, with padding.
expert_tile_1_pad = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 24],
        pad=[0, 1, 2],
        peel=False),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 1 level of tiling, with padding, hoisted.
expert_tile_1_pad_hoist = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 64],
        pad=[0, 1, 2],
        hoist_padding=3,
        peel=False),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 2 levels of tiling, with padding, hoisted.
expert_tile_2_pad_hoist = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 24],
        pad=[],
        peel=False),
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[4, 4, 12],
        pad=[0, 1, 2],
        hoist_padding=6,
        peel=False),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
expert_tile_3_pad_hoist_peel = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 24],
        pad=[],
        peel=False),
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[4, 4, 12],
        pad=[0, 1, 2],
        hoist_padding=6,
        peel=False),
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[2, 3, 7],
        pad=[],
        peel=[0, 1, 2]),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
# Scalarize remaining dynamic dims.
expert_tile_3_pad_hoist_peel_scalarize = TestExpert([
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 24],
        pad=[],
        peel=False),
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[4, 4, 12],
        pad=[0, 1, 2],
        hoist_padding=6,
        peel=False),
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[2, 3, 7],
        pad=[],
        peel=[0, 1, 2]),
    Tile('matmul_on_tensors', 'linalg.matmul', scalarize_dyn_dims=True),
    Vectorize('matmul_on_tensors', 'linalg.matmul')
])

# Fuse, then tile.
expert_fuse_2_tile_1 = TestExpert([
    Fuse('matmul_on_tensors', 'linalg.matmul', tile_sizes=[8, 16, 0]),
    Fuse('matmul_on_tensors', 'linalg.matmul', tile_sizes=[4, 4, 0]),
    Tile('matmul_on_tensors', 'linalg.matmul', tile_sizes=[0, 0, 24]),
    Vectorize('matmul_on_tensors', 'linalg.matmul'),
    Vectorize('matmul_on_tensors', 'linalg.fill')
])

expert_fuse_and_pad = TestExpert([
    Fuse('matmul_on_tensors', 'linalg.matmul', [16, 16, 0]),
    Tile(
        'matmul_on_tensors',
        'linalg.matmul',
        tile_sizes=[8, 8, 32],
        pad=[0, 1, 2],
        hoist_padding=3),
    Vectorize('matmul_on_tensors', 'linalg.matmul'),
    Tile('matmul_on_tensors', 'linalg.fill', tile_sizes=[8, 8]),
    Vectorize('matmul_on_tensors', 'linalg.fill')
])

# TODO: Fix broken tests.
all_experts = [
    expert_no_tiling, expert_tile_1, expert_tile_1_peel_scalarize,
    expert_tile_1_pad, expert_tile_1_pad_hoist, expert_tile_2_pad_hoist,
    expert_tile_3_pad_hoist_peel, expert_tile_3_pad_hoist_peel_scalarize,
    expert_fuse_2_tile_1, expert_fuse_and_pad
]


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[24, 32, 48], [27, 37, 43]]
  for np_type in [np.float32]:
    for problem_sizes in problem_size_list:
      M, N, K = problem_sizes
      print(
          f'\n###############################################################\n'
          f'Problem size {M}x{N}x{K}')
      for expert in all_experts:
        compile_and_test_linalg_matmul(M, N, K, n_iters, np_type, expert, False)


if __name__ == '__main__':
  main()
