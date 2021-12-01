# RUN: %PYTHON %s 2>&1 | FileCheck %s

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *
from .ops import *

import typing as tp

fun_name = 'transpose_4d_on_tensors'
op_name = 'linalg.generic'

def tiling_shuffle_lowering(**kwargs):
  return TileAndDecompose(**kwargs)\
    .then(Vectorize(fun_name, op_name))\
    .then(Bufferize())\
    .then(LowerVectors(transpose_lowering='shuffle'))\
    .then(LowerToLLVM())

expert_transpose_4d_0213 = tiling_shuffle_lowering(
    fun_name=fun_name,
    op_name=op_name,
    tile_sizes=[1, 4, 4, 16],
    peel=[0, 1]).print_ir(after_all=False)

expert_transpose_4d_1302 = tiling_shuffle_lowering(
    fun_name=fun_name,
    op_name=op_name,
    tile_sizes=[1, 0, 4, 4],
    peel=[0, 1]).print_ir(after_all=False)

################################################################################
### Problem instantiations.
################################################################################


# CHECK-NOT: FAILURE
def main():
  n_iters = 25
  keys = ['M', 'N', 'K', 'L']

  # List of problems (sizes, permutation, op, expert strategy).
  problem_list = [
      # [16, 100, 8, 128] -> [16, 8, 100, 128]
      ([16, 100, 8, 128], [0, 2, 1,
                           3], transpose_4d_0213, expert_transpose_4d_0213),
      # [64, 224, 224, 3] -> [224, 3, 64, 224]
      ([64, 224, 224, 3], [1, 3, 0,
                           2], transpose_4d_1302, expert_transpose_4d_1302),
      # [128, 224, 224, 3] -> [224, 3, 224, 128]
      ([128, 224, 224, 3], [1, 3, 0,
                            2], transpose_4d_1302, expert_transpose_4d_1302),
      # [8, 224, 224, 3] -> [224, 3, 8, 224]
      ([8, 224, 224, 3], [1, 3, 0,
                          2], transpose_4d_1302, expert_transpose_4d_1302),
  ]

  def make_size_list(keys: tp.Sequence[str], sizes: tp.Sequence):
    return {k: v for k, v in zip(keys, sizes)}

  for problem in problem_list:
    test_harness(
        lambda s, t: TransposeNDProblem(
            permutation=problem[1], op_builder=problem[2]), [[np.float32] * 2],
        [make_size_list(keys, problem[0])],
        experts=[problem[3]],
        n_iters=n_iters,
        function_name=fun_name)


if __name__ == '__main__':
  main()
