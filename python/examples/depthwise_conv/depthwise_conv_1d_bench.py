# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'depthwise_conv_1d_nwc_wc'
op_name = 'linalg.depthwise_conv_1d_nwc_wc'

################################################################################
# Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
        SingleTilingExpert(
            fun_name=fun_name,
            op_name=op_name,
            #           N  W   C  KW
            # tile_sizes=[1, 4, 16, 3]
            tile_sizes=[4, 4, 16, 3])
    ]
]

################################################################################
# Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'strides', 'dilations']


def make_size_list(keys: Sequence[str], sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}


# CHECK-NOT: FAILURE
def main():
  n_iters = 1000

  # Specify default configuration and parse command line.
  args = test_argparser(
    "depthwise conv 2d benchmark",
    #  N   W   C  KW   st  dil
    default_problem_sizes_list = [
      [8, 16, 32, 3, [1], [1]],
      [8, 16, 32, 3, [1], [2]],
      [8, 16, 32, 3, [2], [1]],
      [8, 16, 32, 3, [2], [2]],
      [8, 16, 32, 3, [2], [3]],
      [8, 16, 32, 3, [3], [2]]
    ],
    default_expert_list = [
      idx for idx, _ in enumerate(all_experts)
    ],
    default_runtime_only_list = [],
    default_spec_list = [])

  test_harness(
      lambda sizes, t: DepthwiseConvolutionProblem(
          'NWC', 'WC', strides=sizes['strides'], dilations=sizes['dilations']),
      [[np.float32] * 3],
      map(lambda sizes: make_size_list(keys, sizes), args.problem_sizes_list),
      [all_experts[idx] for idx in args.expert_list if idx < len(all_experts)],
      n_iters=n_iters,
      function_name=fun_name,
      dump_ir_to_file='/tmp/abcd.mlir',
      dump_obj_to_file='/tmp/abcd.o',
  )


if __name__ == '__main__':
  main()
