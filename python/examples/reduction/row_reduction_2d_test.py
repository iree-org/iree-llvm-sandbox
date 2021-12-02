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
    'row_reduction_2d_on_tensors', 'linalg.generic').print_ir(after_all=False)

expert_fuse_output = TransformationList(transforms=[ExperimentalSplitAndFuseFillOp(
        'row_reduction_2d_on_tensors', 'linalg.generic', tile_sizes=[24, 16])]
        +
        LoweringOnlyExpert(
        'row_reduction_2d_on_tensors', 'linalg.generic').transforms)

all_experts = [expert_no_tiling, expert_fuse_output]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'K']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[48, 16], [49, 17]]

  test_harness(
      lambda s, t: RowReduction2DProblem(), [[np.float32] * 2],
      map(make_size_list, problem_size_list),
      all_experts,
      n_iters=n_iters,
      function_name='row_reduction_2d_on_tensors')


if __name__ == '__main__':
  main()
