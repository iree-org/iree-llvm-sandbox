# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains test to compile fusion examples.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *
from ..core.transform import PrintIR

from .definitions import *

################################################################################
### Expert for running the fusion tests.
################################################################################


# 1 linalg.fill -> linalg.matmul fusion.
def fill_matmul_fusion():
  fun_name = 'matmul'
  op_name = 'linalg.matmul'
  expert = Fuse(fun_name, op_name, tile_sizes=[8, 16, 0],          \
                tile_interchange=[0, 1, 2])                        \
    .then(Pad(fun_name, 'linalg.fill',                             \
              padding_values=[0.0, 0.0]))                      \
    .then(Tile(fun_name, op_name, tile_sizes=[0, 0, 24]))          \
    .then(Pad(fun_name, op_name,                                   \
              padding_values=[0.0, 0.0, 0.0],                \
              pack_paddings=[1, 1, 0]))                            \
    .then(Vectorize(fun_name, '', vectorize_paddings=True))        \
    .then(LoweringOnlyExpert('', ''))
  keys = ['M', 'N', 'K']
  n_iters = 1
  problem_size_list = [[24, 32, 48], [27, 37, 43]]
  test_harness(lambda s, t: MatmulProblem(), [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               [expert.print_ir(after_all=False, at_begin=False, llvm=False)],
               n_iters=n_iters,
               function_name=fun_name,
               zero_at_each_iteration=True)


def fill_matmul_bias_add_fusion():
  fun_name = 'matmul_bias_add'
  op_name = 'linalg.generic'
  # FIXME: Cannot pad and vectorize a generic consuming a for loop output.
  expert = Fuse(fun_name, op_name, tile_sizes=[8, 16],               \
                tile_interchange=[0, 1])                             \
      .then(Pad(fun_name, 'linalg.fill',                             \
                padding_values=[0.0, 0.0]))                      \
      .then(Tile(fun_name, 'linalg.matmul', tile_sizes=[0, 0, 24]))  \
      .then(Pad(fun_name, 'linalg.matmul',                           \
                padding_values=[0., 0., 0.],       \
                pack_paddings=[1, 1, 0]))                            \
      .then(LoweringOnlyExpert('', ''))

  keys = ['M', 'N', 'K']
  n_iters = 1
  problem_size_list = [[24, 32, 48], [27, 37, 43]]
  test_harness(lambda s, t: MatmulBiasAddProblem(), [[np.float32] * 4],
               test_sizes(keys, problem_size_list),
               [expert.print_ir(after_all=False, at_begin=False, llvm=False)],
               n_iters=n_iters,
               function_name=fun_name,
               zero_at_each_iteration=True)


# CHECK-NOT: FAILURE
def main():
  fill_matmul_fusion()
  fill_matmul_bias_add_fusion()


if __name__ == '__main__':
  main()
