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

all_experts = [
    SingleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        #           N  W   C  KW  F
        tile_sizes=[1, 8, 32, 1, 8],
        tile_interchange=[],
        peel=[],
        pad=False,
        pack_paddings=[],
        hoist_paddings=[],
        # kwargs passed down to LowerVectors.
        # TODO: better composition of experts.
        transpose_lowering='shuffle',
        print_ir_after_all=False)
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'F', 'strides', 'dilations']


def make_size_list(keys: tp.Sequence[str], sizes: tp.Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  #   N   W   C  KW   F  st  dil
  problem_size_list = [\
     [8, 16, 32,  3, 64,  [1],  [1]], \
     [8, 16, 32,  3, 64,  [1],  [2]], \
     [8, 16, 32,  3, 64,  [2],  [1]], \
     [8, 16, 32,  3, 64,  [2],  [2]], \
     [8, 16, 32,  3, 64,  [2],  [3]], \
     [8, 16, 32,  3, 64,  [3],  [2]],  \
  ]

  test_harness(
      lambda sizes, types: ConvolutionProblem(
          'NWC', 'WCF', strides=sizes['strides'], dilations=sizes['dilations']),
      [[np.float32] * 3],
      [make_size_list(keys, sizes) for sizes in problem_size_list],
      all_experts,
      n_iters=n_iters,
      function_name=fun_name)


if __name__ == '__main__':
  main()
