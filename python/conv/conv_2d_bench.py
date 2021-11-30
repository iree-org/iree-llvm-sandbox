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
### Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False) for e in [
        TileAndDecompose(
            fun_name=fun_name,
            op_name=op_name,
            #           N  H  W  C  KH  KW  F
            tile_sizes=[1, 1, 8, 32, 1, 1, 8]) + \
          Vectorize(fun_name, op_name) + \
          Bufferize() + \
          StagedVectorLowering(transpose_lowering='shuffle') +\
          LowerToLLVM()
    ]
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'H', 'W', 'C', 'KH', 'KW', 'F', 'strides', 'dilations']


def make_size_list(keys: tp.Sequence[str], sizes: tp.Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  #   N   H   W   C  KH  KW   F     st      dil
  problem_size_list = [\
     [8, 16, 16, 32,  3,  3, 64, [1, 1], [1, 1]], \
     [8, 16, 16, 32,  3,  3, 64, [1, 2], [1, 2]], \
     [8, 16, 16, 32,  3,  3, 64, [2, 1], [1, 2]], \
     [8, 16, 16, 32,  3,  3, 64, [2, 2], [2, 2]],  \
     [8, 16, 16, 32,  3,  3, 64, [2, 3], [3, 2]],  \
     [8, 16, 16, 32,  3,  3, 64, [3, 2], [2, 3]],  \
  ]

  test_harness(
      lambda sizes, types: ConvolutionProblem(
          'NHWC',
          'HWCF',
          strides=sizes['strides'],
          dilations=sizes['dilations']), [[np.float32] * 3],
      [make_size_list(keys, sizes) for sizes in problem_size_list],
      all_experts,
      n_iters=n_iters,
      function_name=fun_name)


if __name__ == '__main__':
  main()
