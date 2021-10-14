# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from .util import *

all_experts = [
    SingleTilingExpert(
        sizes=[8, 16, 32],
        interchange=[0, 1, 2],
        peel=False,
        pad=True,
        pack_padding=[0, 1, 2],
        hoist_padding=[2, 3, 0])
]

# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  problem_size_list = [[128, 192, 256], [104, 96, 108]]
  for np_type in [np.float32]:
    for problem_sizes in problem_size_list:
      M, N, K = problem_sizes
      # Init printing.
      print(
          f'\n###############################################################\n'
          f'Problem size {M}x{N}x{K}')
      print('MLIR Codegen')
      for expert in all_experts:
        compile_and_test_linalg_matmul(M, N, K, n_iters, np_type, expert)
      # For single-threaded apples-to-apples comparisons, run with:
      # MKL_NUM_THREADS=1 ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
      import os
      if os.environ.get('BENCHMARK_NUMPY'):
        print('Numpy')
        test_numpy_matmul(M, N, K, n_iters, np_type)
      if os.environ.get('BENCHMARK_TORCH'):
        print('Torch')
        test_torch_matmul(M, N, K, n_iters, np_type, 1)


if __name__ == '__main__':
  main()
