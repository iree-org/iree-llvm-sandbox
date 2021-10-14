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

all_experts = [expert_no_tiling]


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[24, 32], [27, 37]]
  for np_type in [np.float32]:
    for problem_sizes in problem_size_list:
      M, N = problem_sizes
      print(
          f'\n###############################################################\n'
          f'Problem size {M}x{N}')
      for expert in all_experts:
        compile_and_test_linalg_reduction(M, N, n_iters, np_type, expert, False)


if __name__ == '__main__':
  main()
