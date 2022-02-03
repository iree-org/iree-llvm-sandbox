# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

import typing as tp

fun_name = 'matmul'
op_name = 'linalg.generic'

################################################################################
# Compilation strategies.
################################################################################

all_experts = [
    e.print_pipeline(before_all=False) for e in [                             \
        e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [    \
            LinalgExtTile(fun_name, op_name, tile_sizes=[4])
              .then(LinalgExtTileToInParallel(fun_name, op_name))
              .then(Bufferize())
              .then(LinalgExtInParallelToSequentialFor(fun_name, op_name))
              .then(LowerToLLVM()),
            LinalgExtTile(fun_name, op_name, tile_sizes=[4])
              .then(LinalgExtTileToInParallel(fun_name, op_name))
              .then(Bufferize())
              .then(LinalgExtInParallelToAsync(fun_name, op_name))
              .then(LowerToLLVM()),
            LinalgExtTile(fun_name, op_name, tile_sizes=[16])
              .then(LinalgExtTileToInParallel(fun_name, op_name))
              .then(Tile(fun_name,
                         op_name,
                         tile_sizes=[12, 32, 16],
                         pad=True,
                         pack_paddings=[1, 1, 0],
                         hoist_paddings=[2, 3, 0]))
              .then(Vectorize(fun_name, ''))
              .then(LoweringOnlyExpert(fun_name, op_name)),
            LinalgExtTile(fun_name, op_name, tile_sizes=[0, 16])
              .then(LinalgExtTileToInParallel(fun_name, op_name))
              .then(Tile(fun_name,
                         op_name,
                         tile_sizes=[12, 32, 16],
                         pad=True,
                         pack_paddings=[1, 1, 0],
                         hoist_paddings=[2, 3, 0]))
              .then(Vectorize(fun_name, ''))
              .then(LoweringOnlyExpert(fun_name, op_name)),
        ]
    ]
]

################################################################################
# Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[100, 200, 300]]
  test_harness(lambda s, t: EinsumProblem('mk,kn', 'mnk', 2),
               [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name='matmul')


if __name__ == '__main__':
  main()
