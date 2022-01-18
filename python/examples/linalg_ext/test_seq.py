# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

import typing as tp

################################################################################
# Compilation strategies.
################################################################################

expert_linalg_ext_tile = TransformationList(transforms=[
    LinalgExtTile('matmul_on_tensors', 'linalg.generic', tile_sizes=[2]),
    LinalgExtTileToSequentialFor('matmul_on_tensors', 'linalg.generic'),
    Vectorize('matmul_on_tensors', 'linalg.generic'),
    Bufferize(),
    LowerToLLVM(),
])

all_experts = [
    e.print_ir(after_all=False, llvm=False) for e in [expert_linalg_ext_tile]
]

################################################################################
# Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[3, 5, 7]]
  test_harness(lambda s, t: EinsumProblem('mk,kn', 'mnk', 2), [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name='matmul')


if __name__ == '__main__':
  main()
