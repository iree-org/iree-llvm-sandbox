# RUN: %PYTHON %s 2>&1 | FileCheck %s

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

import typing as tp

fun_name = 'transpose_4d_on_tensors'
op_name = 'linalg.generic'


def tiling_shuffle_lowering(**kwargs):
  return TileAndDecompose(**kwargs)                                           \
    .then(Vectorize(fun_name, op_name))                                       \
    .then(Bufferize())                                                        \
    .then(LowerVectors(transpose_lowering='shuffle'))                         \
    .then(LowerToLLVM())


expert_transpose_4d_0213 = tiling_shuffle_lowering(                           \
    fun_name=fun_name,                                                        \
    op_name=op_name,                                                          \
    tile_sizes=[1, 4, 4, 16],                                                 \
    peel=[0, 1])                                                              \
  .print_ir(after_all=False)

expert_transpose_4d_1302 = tiling_shuffle_lowering(                           \
    fun_name=fun_name,                                                        \
    op_name=op_name,                                                          \
    tile_sizes=[1, 0, 4, 4],                                                  \
    peel=[0, 1])                                                              \
  .print_ir(after_all=False)

################################################################################
### Problem instantiations.
################################################################################


# CHECK-NOT: FAILURE
def main():
  n_iters = 25
  keys = ['k', 'l', 'm', 'n']

  # List of problems (sizes, einsum spec, expert strategy).
  problem_list = [
      # [16, 100, 8, 128] -> [16, 8, 100, 128]
      ([16, 8, 100, 128], \
        'kmln->klmn',
        expert_transpose_4d_0213),
      # [64, 224, 224, 3] -> [224, 3, 64, 224]
      ([224, 3, 64, 224], \
        'mknl->klmn',
        expert_transpose_4d_1302),
      # [128, 224, 224, 3] -> [224, 3, 224, 128]
      ([224, 3, 224, 128], \
        'mknl->klmn',
        expert_transpose_4d_1302),
      # [8, 224, 224, 3] -> [224, 3, 8, 224]
      ([224, 3, 8, 224], \
        'mknl->klmn',
        expert_transpose_4d_1302),
  ]

  for problem in problem_list:
    test_harness(lambda s, t: EinsumProblem(problem[1], 0),
                 [[np.float32] * 2],
                 test_sizes(keys, [problem[0]]), [problem[2]],
                 n_iters=n_iters,
                 function_name=fun_name)


if __name__ == '__main__':
  main()
