# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from numpy import transpose
from mlir.sandbox.experts import *
from mlir.sandbox.harness import *
from mlir.sandbox.transforms import *

from .definitions import *

import typing as tp

fun_name = 'conv_2d_nhwc_hwcf_main'
op_name = 'linalg.conv_2d_nhwc_hwcf'

################################################################################
# Compilation strategies.
################################################################################

# Note: `\` char at the end of next line prevents formatter reflows, keep it.
all_names = [ \
    "SingleTiling3DPeel",
    "SingleTiling3DPad",
    "DoubleTile3DPeel",
    "DoubleTile3DPad",
            ]

all_experts = [ \
    Tile(fun_name=fun_name,
          op_name=op_name,
          #           N  H  W  C  KH  KW  F
          tile_sizes=[1, 1, 8, 32, 1, 1, 8],
          peel=[0, 1, 2, 3, 4, 5, 6])
      .then(DecomposeToLowerDimensionalNamedOp())
      .then(Vectorize(fun_name, ''))
      .then(LoweringOnlyExpert('', '', transpose_lowering='shuffle')),
    Tile(fun_name=fun_name,
          op_name=op_name,
          #           N  H  W  C  KH  KW  F
          tile_sizes=[1, 1, 8, 32, 1, 1, 8])
      .then(Pad(fun_name=fun_name,
                op_name=op_name,
                padding_values=[0.0, 0.0, 0.0],
                padding_dimensions=[0, 1, 2, 3, 4, 5, 6],
                hoist_paddings=[5, 0, 0]))
      .then(DecomposeToLowerDimensionalNamedOp())
      .then(Vectorize(fun_name, ''))
      .then(LoweringOnlyExpert('', '', transpose_lowering='shuffle')),
    Tile(fun_name,
          op_name,
          #           N  H  W  C  KH  KW  F
          tile_sizes=[1, 32, 32, 32, 1, 3, 64],
          peel=[0, 1, 2, 3, 4, 5, 6])
      .then(Tile(fun_name,
                  op_name,
                  tile_sizes=[1, 1, 8, 32, 1, 1, 8],
                  peel=[0, 1, 2, 3, 4, 5, 6]))
      .then(DecomposeToLowerDimensionalNamedOp())
      .then(Vectorize(fun_name, ''))
      .then(LoweringOnlyExpert('', '')),
    Tile(fun_name,
          op_name,
          #           N  H  W  C  KH  KW  F
          tile_sizes=[1, 32, 32, 32, 3, 3, 64])
      .then(Tile(fun_name,
                  op_name,
                  tile_sizes=[1, 1, 8, 32, 1, 1, 8]))
      .then(Pad(fun_name,
                op_name,
                padding_values=[0.0, 0.0, 0.0],
                padding_dimensions=[0, 1, 2, 3, 4, 5, 6],
                pack_paddings=[1, 0, 0],
                hoist_paddings=[4, 0, 0]))
      .then(DecomposeToLowerDimensionalNamedOp())
      .then(Vectorize(fun_name, ''))
      .then(LoweringOnlyExpert(fun_name,
                                op_name,
                                split_transfers='none',
                                transpose_lowering='shuffle',
                                unroll_vector_transfers=False)),
]

################################################################################
# Problem instantiation
################################################################################

keys = ['N', 'H', 'W', 'C', 'KH', 'KW', 'F', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
      "conv 2d benchmark",
      default_n_iters=100,
      #  N   H   W   C  KH  KW   F     st      dil
      default_problem_sizes_list=[
          [8, 16, 16, 32, 3, 3, 64, [1, 1], [1, 1]],
          [8, 16, 16, 32, 3, 3, 64, [1, 2], [1, 2]],
          [8, 16, 16, 32, 3, 3, 64, [2, 1], [1, 2]],
          [8, 16, 16, 32, 3, 3, 64, [2, 2], [2, 2]],
      ],
      default_expert_list=all_names,
      default_dynamic_at_compile_time_list=[ \
        []  # case 1: static at compile time
      ],
      default_spec_list=[])

  for dynamic_at_compile_time in args.dynamic_at_compile_time_list:
    test_harness(lambda sizes, types: ConvolutionProblem(
        'NHWC', 'HWCF', strides=sizes['strides'], dilations=sizes['dilations']),
                 [[np.float32] * 3],
                 test_sizes(keys, args.problem_sizes_list),
                 test_experts(all_experts, all_names, args.expert_list),
                 n_iters=args.n_iters,
                 function_name=fun_name,
                 dump_ir_to_file='/tmp/abcd.mlir',
                 dump_obj_to_file='/tmp/abcd.o',
                 dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
