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
    'column_reduction_2d_on_tensors',
    'linalg.generic')\
  .print_ir(after_all=False)

all_experts = [expert_no_tiling]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'K']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[48, 16], [49, 17]]

  test_harness(lambda s, t: ColumnReduction2DProblem(), [[np.float32] * 2],
               test_sizes(keys, problem_size_list),
               test_experts(all_experts),
               n_iters=n_iters,
               function_name='column_reduction_2d_on_tensors')


if __name__ == '__main__':
  main()
