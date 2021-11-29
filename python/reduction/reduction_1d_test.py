# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.problem_definition import *
from ..core.transforms import *

from .definitions import *

################################################################################
### Compilation strategies.
################################################################################

# No tiling.
expert_no_tiling = LoweringOnlyExpert(
    'reduction_1d_on_tensors', 'linalg.generic').print_ir(after_all=False)

all_experts = [expert_no_tiling]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[48], [49]]

  test_harness(
      lambda s, t: Reduction1DProblem(), [[np.float32] * 2],
      map(make_size_list, problem_size_list),
      all_experts,
      n_iters=n_iters,
      function_name='reduction_1d_on_tensors')


if __name__ == '__main__':
  main()
