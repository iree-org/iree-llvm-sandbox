# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

import typing as tp

fun_name = 'conv_3d_ndhwc_dhwcf_main'
op_name = 'linalg.conv_3d_ndhwc_dhwcf'

################################################################################
### Compilation strategies.
################################################################################

# Note: `\` char at the end of next line prevents formatter reflows, keep it.
all_names = [ \
  "Conv3DExpert"
            ]

all_experts = [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [ \
        Tile(fun_name,
             op_name,
             #           N  D  H  W  C   KH KW F
             tile_sizes=[1, 2, 1, 8, 32, 1, 1, 8])
        .then(Vectorize(fun_name, ''))
        .then(LoweringOnlyExpert(fun_name, op_name)),
    ]
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'D', 'H', 'W', 'C', 'KD', 'KH', 'KW', 'F', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
      "conv 3d benchmark",
      default_n_iters=1,
      #  N  D   H   W   C  KD  KH  KW   F     st      dil
      default_problem_sizes_list=[
          [8, 4, 16, 16, 32, 3, 3, 3, 64, [1, 1, 1], [1, 1, 1]],
          [8, 4, 16, 16, 32, 3, 3, 3, 64, [1, 1, 2], [1, 1, 2]],
          [8, 4, 16, 16, 32, 3, 3, 3, 64, [1, 2, 1], [1, 1, 2]],
          [8, 4, 16, 16, 32, 3, 3, 3, 64, [1, 2, 2], [1, 2, 2]],
      ],
      default_expert_list=all_names,
      default_dynamic_at_compile_time_list=[],
      default_spec_list=[])

  test_harness(lambda sizes, types: ConvolutionProblem(
      'NDHWC', 'DHWCF', strides=sizes['strides'], dilations=sizes['dilations']),
               [[np.float32] * 3],
               test_sizes(keys, args.problem_sizes_list),
               test_experts(all_experts, all_names, args.expert_list),
               n_iters=args.n_iters,
               function_name=fun_name,
               dump_data_to_file=args.dump_data,
               backends=['dialect'])


if __name__ == '__main__':
  main()
