# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *
from ..core.utils import *

from ..contraction.definitions import EinsumProblem

from typing import List

fun_name = 'copy_1d'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

# Before bufferization, the IR only has a tensor.extract_slice /
#   tensor.insert_slice pair.
# Bufferization then properly introduces copy ops (implemented with
# linalg.generic)
# We want to make more these copies more efficient.
# In the case of a single copy benchmark it is the one true thing to optimize.
all_experts = [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [         \
      Tile(fun_name=fun_name,
            op_name=op_name,
            tile_sizes=[16],
            peel=[0])
      .then(Bufferize())
      .then(Vectorize(fun_name=fun_name, op_name=''))
      .then(LowerVectors())
      .then(LowerToLLVM())
    ]
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['n']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
      "copy 1d benchmark",
      default_n_iters=100,
      default_problem_sizes_list=[
          [100 * 32],
          [100 * 272],
          [200 * 272],
          [300 * 272],
          [400 * 272],
          [500 * 272],
          [600 * 272],
          [700 * 272],
          [800 * 272],
          [900 * 272],
          [1000 * 272],
          [10000 * 272],
          [20000 * 272],
          [30000 * 272],
          [300000 * 272],
      ],
      default_expert_list=["Default"],
      default_dynamic_at_compile_time_list=[
          [],  # case 1: static at compile time
      ],
      default_spec_list=['n->n'],
  )

  for dynamic_at_compile_time in args.dynamic_at_compile_time_list:
    for spec in args.spec_list:
      test_harness(\
          lambda s, t: EinsumProblem('n->n', 'n', 0),
          [[np.float32] * 2],
          test_sizes(keys, args.problem_sizes_list),
          test_experts(all_experts, ["Default"], args.expert_list),
          n_iters=args.n_iters,
          dynamic_at_compile_time_sizes=set(
              dynamic_at_compile_time).intersection(keys),
          function_name=fun_name,
          dump_ir_to_file='/tmp/abc.mlir',
          dump_obj_to_file='/tmp/abc.o',
          dump_data_to_file=args.dump_data,
          backends=['dialect'])


if __name__ == '__main__':
  main()
