# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

import typing as tp

fun_name = 'conv_1d_nwc_wcf_main'
op_name = 'linalg.conv_1d_nwc_wcf'

################################################################################
### Compilation strategies.
################################################################################

all_names = [
  "Conv1DExpert"
]
all_experts = [
    e.print_ir(after_all=False) for e in [
        TileAndDecompose(
            fun_name,
            op_name,
            #           N  W   C  KW  F
            tile_sizes=[1, 8, 32, 1, 8],
            pad=True,
            hoist_paddings=[4, 0, 0]) +\
          Vectorize(fun_name, op_name) +\
          Bufferize() +\
          LowerVectors(transpose_lowering='shuffle') +\
          LowerToLLVM()
    ]
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'F', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1000

  # Specify default configuration and parse command line.
  args = test_argparser(
    "conv 1d benchmark",
    #  N   W   C  KW   F  st  dil
    default_problem_sizes_list = [
      [8, 16, 32,  3, 64,  [1],  [1]],
      [8, 16, 32,  3, 64,  [1],  [2]],
      [8, 16, 32,  3, 64,  [2],  [1]],
      [8, 16, 32,  3, 64,  [2],  [2]],
      [8, 16, 32,  3, 64,  [2],  [3]],
      [8, 16, 32,  3, 64,  [3],  [2]],
    ],
    default_expert_list = all_names,
    default_dynamic_at_compile_time_list = [],
    default_spec_list = [])

  test_harness(lambda sizes, types: ConvolutionProblem(
      'NWC', 'WCF', strides=sizes['strides'], dilations=sizes['dilations']),
               [[np.float32] * 3],
               test_sizes(keys, args.problem_sizes_list),
               test_experts(all_experts, all_names, args.expert_list),
               n_iters=n_iters,
               function_name=fun_name,
               plot_path=args.plot_path)


if __name__ == '__main__':
  main()
