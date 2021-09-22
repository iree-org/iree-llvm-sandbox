# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from experts import *
from linalg_matmul import *


# Expert compiler that applies a single level of tiling.
class SingleTilingExpert(Expert):
  variables = {
      'sizes': TilingSizesVariable,
      'pad': BoolVariable,
      'peel': BoolVariable,
      'hoist_padding': HoistPaddingVariable,
  }

  def transforms(self) -> List[Transform]:
    v = self.assignments
    return [
        Tile(
            'matmul_on_tensors',
            'linalg.matmul',
            tile_sizes=v.sizes,
            pad=v.pad,
            peel=v.peel,
            hoist_padding=v.hoist_padding),
        Vectorize('matmul_on_tensors', 'linalg.matmul'),
        Bufferize(),
        LowerToLLVM(),
    ]


all_experts = [
    SingleTilingExpert(
        sizes=[52, 48, 54], pad=True, peel=False, hoist_padding=3)
]

# CHECK-NOT: FAILURE
n_iters = 10
problem_size_list = [[128, 192, 256], [104, 96, 108]]
for np_type in [np.float32]:
  for problem_sizes in problem_size_list:
    M, N, K = problem_sizes
    # Init printing.
    print(f'\n###############################################################\n'
          f'Problem size {M}x{N}x{K}')
    for expert in all_experts:
      compile_and_test_linalg_matmul(M, N, K, n_iters, np_type, expert)
    # For single-threaded apples-to-apples comparisons, run with:
    # MKL_NUM_THREADS=1 ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
    import os
    if os.environ.get('BENCHMARK_NUMPY'):
      test_numpy_matmul(M, N, K, n_iters, np_type)
    if os.environ.get('BENCHMARK_TORCH'):
      test_torch_matmul(M, N, K, n_iters, np_type, 1)
