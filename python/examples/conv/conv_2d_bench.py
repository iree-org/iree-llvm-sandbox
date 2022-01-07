# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

import typing as tp

fun_name = 'conv_2d_nhwc_hwcf_main'
op_name = 'linalg.conv_2d_nhwc_hwcf'

################################################################################
# Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
        TileAndDecompose(
            fun_name=fun_name,
            op_name=op_name,
            #           N  H  W  C  KH  KW  F
            tile_sizes=[1, 1, 8, 32, 1, 1, 8],
            pad=True,
            hoist_paddings=[5, 0, 0]) + \
        Vectorize(fun_name, "") + \
        Bufferize() + \
        LowerVectors(transpose_lowering='shuffle') +\
        LowerToLLVM()
    ]
]

################################################################################
# Problem instantiation
################################################################################

keys = ['N', 'H', 'W', 'C', 'KH', 'KW', 'F', 'strides', 'dilations']


def make_size_list(keys: tp.Sequence[str], sizes: tp.Sequence):
  return {k: v for k, v in zip(keys, sizes)}


# CHECK-NOT: FAILURE
def main():
  n_iters = 100

  # Specify default configuration and parse command line.
  args = test_argparser(
    "conv 2d benchmark",
    #  N   H   W   C  KH  KW   F     st      dil
    default_problem_sizes_list = [
      [8, 16, 16, 32, 3, 3, 64, [1, 1], [1, 1]],
      [8, 16, 16, 32, 3, 3, 64, [1, 2], [1, 2]],
      [8, 16, 16, 32, 3, 3, 64, [2, 1], [1, 2]],
      [8, 16, 16, 32, 3, 3, 64, [2, 2], [2, 2]],
      [8, 16, 16, 32, 3, 3, 64, [2, 3], [3, 2]],
      [8, 16, 16, 32, 3, 3, 64, [3, 2], [2, 3]],
    ],
    default_expert_list = [
      idx for idx, _ in enumerate(all_experts)
    ],
    default_dynamic_at_compile_time_list = [],
    default_spec_list = [])

  test_harness(
      lambda sizes, types: ConvolutionProblem('NHWC',
                                              'HWCF',
                                              strides=sizes['strides'],
                                              dilations=sizes['dilations']),
      [[np.float32] * 3],
      [make_size_list(keys, sizes) for sizes in args.problem_sizes_list],
      [all_experts[idx] for idx in args.expert_list if idx < len(all_experts)],
      n_iters=n_iters,
      function_name=fun_name,
      dump_ir_to_file='/tmp/abcd.mlir',
      dump_obj_to_file='/tmp/abcd.o',
  )


if __name__ == '__main__':
  main()
