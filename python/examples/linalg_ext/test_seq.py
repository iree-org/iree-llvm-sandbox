# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

from mlir.iree_sandbox import *
from mlir.dialects.iree_linalg_ext import *
from mlir.dialects.linalg_transform import *

import typing as tp

fun_name = 'matmul'
op_name = 'linalg.generic'

################################################################################
# Compilation strategies.
################################################################################

expert_iree_linalg_ext_tile = [                                 \
    LinalgExtTile(fun_name, op_name, tile_sizes=[2])       \
    .then(LinalgExtTileToScfFor(fun_name)) \
    .then(Vectorize(fun_name, ''))\
    .then(LoweringOnlyExpert(fun_name, op_name, transpose_avx2_lowering=True))\
]
all_experts = expert_iree_linalg_ext_tile

################################################################################
# Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[3, 5, 7]]
  test_harness(lambda s, t: EinsumProblem('mk,kn', 'mnk', 2),
               [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name=fun_name)


if __name__ == '__main__':
  main()
