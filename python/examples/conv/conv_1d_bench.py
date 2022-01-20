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

all_names = [ \
    "SingleTiling3DPeel",
    "SingleTiling3DPad",
    "DoubleTile3DPeel",
    "DoubleTile3DPad",
            ]

all_experts = [
    e.print_ir(after_all=False, llvm=False) for e in [ \
        SingleTilingExpert(
            fun_name,
            op_name,
            #           N  W   C  KW  F
            tile_sizes=[1, 8, 32, 1, 8],
            peel=[0, 1, 2, 3, 4])
          .then(Vectorize(fun_name, op_name))
          .then(Bufferize())
          .then(LowerVectors())
          .then(LowerToLLVM()),
        SingleTilingExpert(
            fun_name,
            op_name,
            #           N  W   C  KW  F
            tile_sizes=[1, 8, 32, 1, 8],
            pad=True,
            pack_paddings=[1, 1, 0],
            hoist_paddings=[3, 0, 0])
          .then(Vectorize(fun_name, op_name))
          .then(Bufferize())
          .then(LowerVectors())
          .then(LowerToLLVM()),
        DoubleTilingExpert(fun_name,
                           op_name,
                           #            N    W    C KW    F
                           tile_sizes1=[1,  32, 128, 3,  32],
                           tile_sizes2=[1,   8,  32, 1,   8],
                           peel2=[0, 1, 2, 3, 4])
          .then(Vectorize(fun_name, op_name))
          .then(Bufferize())
          .then(LowerVectors())
          .then(LowerToLLVM()),
        DoubleTilingExpert(fun_name,
                           op_name,
                           #            N    W    C KW    F
                           tile_sizes1=[1,  32, 128, 3,  32],
                           tile_sizes2=[1,   8,  32, 1,   8],
                           pad2=True,
                           pack_paddings2=[1, 1, 0],
                           hoist_paddings2=[3, 0, 0])
          .then(Vectorize(fun_name, op_name))
          .then(Bufferize())
          .then(LowerVectors(split_transfers='vector-transfers'))
          .then(LowerToLLVM()),
    ]
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'F', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
      "conv 1d benchmark",
      default_n_iters=100,
      #  N   W   C  KW   F  st  dil
      default_problem_sizes_list=[
          [8, 16, 32, 3, 64, [1], [1]],
          [8, 16, 32, 3, 64, [1], [2]],
          [8, 16, 32, 3, 64, [2], [1]],
          [8, 16, 32, 3, 64, [2], [2]],
          [8, 16, 32, 3, 64, [2], [3]],
          [8, 16, 32, 3, 64, [3], [2]],
      ],
      default_expert_list=all_names,
      default_dynamic_at_compile_time_list=[],
      default_spec_list=[])

  test_harness(lambda sizes, types: ConvolutionProblem(
      'NWC', 'WCF', strides=sizes['strides'], dilations=sizes['dilations']),
               [[np.float32] * 3],
               test_sizes(keys, args.problem_sizes_list),
               test_experts(all_experts, all_names, args.expert_list),
               n_iters=args.n_iters,
               function_name=fun_name,
               dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
