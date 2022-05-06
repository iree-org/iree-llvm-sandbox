# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains test to compile fusion examples.

from mlir.sandbox.experts import *
from mlir.sandbox.harness import *
from mlir.sandbox.transforms import *
from mlir.sandbox.transform import PrintIR

from .definitions import *

################################################################################
### Expert for running the fusion tests.
################################################################################

def fill_matmul_fusion():
  fun_name = 'matmul'
  op_name = 'linalg.matmul'
  expert = Tile(fun_name, op_name, tile_sizes=[12, 32, 16],        \
                tile_interchange=[0, 1, 2])                        \
    .then(Pad(fun_name, op_name,                                   \
              padding_values=[0.0, 0.0, 0.0],                      \
              padding_dimensions=[0, 1, 2],                        \
              pack_paddings=[1, 1, 0],                             \
              hoist_paddings=[2, 3, 0],                            \
              transpose_paddings=[[0, 1], [0, 1], [0, 1]]))        \
    .then(Vectorize(fun_name, '', vectorize_paddings=True))        \
    .then(Bufferize()) \
    .then(LowerVectors(split_transfers='none')) \
    .then(LowerToLLVM())

  keys = ['M', 'N', 'K']
  n_iters = 1000
  problem_size_list = [[61, 33, 54]]
  test_harness(lambda s, t: MatmulProblem(), [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               [expert.print_ir(after_all=False, at_begin=False, llvm=False)],
               n_iters=n_iters,
               function_name=fun_name,
               zero_at_each_iteration=False)


def fill_matmul_bias_add_fusion():
  fun_name = 'matmul_bias_add'
  op_name = 'linalg.generic'
  # FIXME: Split padding and hoist padding.
  expert = Fuse(fun_name, op_name, tile_sizes=[12, 32],              \
                tile_interchange=[0, 1])                             \
      .then(Pad(fun_name, 'linalg.fill',                             \
                padding_values=[0.0, 0.0],                           \
                padding_dimensions=[0, 1]))                          \
      .then(Pad(fun_name, 'linalg.matmul',                           \
                padding_values=[0.0, 0.0, 0.0],                      \
                padding_dimensions=[0, 1]))                          \
      .then(Pad(fun_name, op_name,                                   \
                padding_values=[0., 0., 0.],                         \
                padding_dimensions=[0, 1],                           \
                pack_paddings=[0, 0],                                \
                hoist_paddings=[2, 0]))                              \
      .then(Tile(fun_name, 'linalg.matmul', tile_sizes=[0, 0, 16]))  \
      .then(Pad(fun_name, 'linalg.matmul',                           \
                padding_values=[0., 0., 0.],                         \
                padding_dimensions=[2],                              \
                pack_paddings=[1, 1, 0]))                            \
      .then(Pad(fun_name, 'linalg.matmul',                           \
                hoist_paddings=[2, 3, 0],                            \
                transpose_paddings=[[0, 1], [0, 1], [0, 1]]))        \
      .then(Vectorize(fun_name, '', vectorize_paddings=True))        \
      .then(Bufferize()) \
      .then(LowerVectors(split_transfers='none')) \
      .then(LowerToLLVM())

  keys = ['M', 'N', 'K']
  n_iters = 1000
  problem_size_list = [[61, 33, 54]]
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
