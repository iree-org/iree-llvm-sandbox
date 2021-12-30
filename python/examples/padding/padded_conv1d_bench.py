# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'padded_conv1d_nwc_wcf_main'
op_name = 'linalg.conv_1d_nwc_wcf'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    SingleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        #           N  W   C  KW  F
        tile_sizes=[1, 8, 32, 1, 8]).print_ir(at_begin=True, after_all=False)
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'F', 'WpadL', 'WpadR', 'stride', 'dilation']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  #   N   W   C  KW   F  WpadL WpadR stride dilation
  problem_size_list = [\
     [8, 16, 32,  3, 64,     0,    1,     1,       1], \
  ]

  test_harness(lambda sizes, t: Padded_Conv1d_NWC_WCF_Problem(
      WpadL=sizes['WpadL'],
      WpadR=sizes['WpadR'],
      stride=sizes['stride'],
      dilation=sizes['dilation']), [[np.float32] * 3],
               map(make_size_list, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name=fun_name)


if __name__ == '__main__':
  main()
