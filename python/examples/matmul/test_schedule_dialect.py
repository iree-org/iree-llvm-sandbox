# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

import mlir.ir as ir
import mlir.dialects.iree_linalg_transform as transform

from ..core.experts import *
from ..core.harness import *
from ..core.pdl_utils import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

# CHECK-NOT: FAILURE


def add_tile_schedule(module):
  tile_sizes = [6, 16, 1]
  tile_d_x_x_by_1 = True
  tile_s_d_x_by_0_1 = False

  #                  M=A.0   N=B.1   K=A.1
  dimM, dimN, dimK = [0, 0], [1, 1], [0, 1]
  isa_linalg_matmul = match_op_with_sizes_multiple_of(
      module, equivalent_op_name='linalg.matmul')
  isa_linalg_matmul_d_x_x = match_op_with_dynamic_or_static_sizes(
      module,
      equivalent_op_name='linalg.matmul',
      dynamic_spec_list=['d'],
      op_dim_spec_list=[dimM])
  isa_linalg_matmul_s_d_x = match_op_with_dynamic_or_static_sizes(
      module,
      equivalent_op_name='linalg.matmul',
      dynamic_spec_list=['s', 'd'],
      op_dim_spec_list=[dimM, dimN])
  isa_linalg_matmul_s_s_d = match_op_with_dynamic_or_static_sizes(
      module,
      equivalent_op_name='linalg.matmul',
      dynamic_spec_list=['s', 's', 'd'],
      op_dim_spec_list=[dimM, dimN, dimK])
  isa_linalg_matmul_s_s_s = match_op_with_dynamic_or_static_sizes(
      module,
      equivalent_op_name='linalg.matmul',
      dynamic_spec_list=['s', 's', 's'],
      op_dim_spec_list=[dimM, dimN, dimK])
  with InsertionPoint(module.body):
    sequence = transform.SequenceOp()
    with ir.InsertionPoint(sequence.body.blocks[0]):
      matched = transform.MatchOp(isa_linalg_matmul)
      transform.TileOp(matched, sizes=tile_sizes, peel=[0, 1, 2])

      if tile_d_x_x_by_1:
        matched = transform.MatchOp(isa_linalg_matmul_d_x_x)
        transform.TileOp(matched, sizes=[1])

      if tile_s_d_x_by_0_1:
        matched = transform.MatchOp(isa_linalg_matmul_s_d_x)
        transform.TileOp(matched, sizes=[0, 1])

      matched = transform.MatchOp(isa_linalg_matmul_s_s_s)
      transform.VectorizeOp(matched)

      transform.BufferizeOp()
      for i in range(7):
        transform.LowerVectorsOp(stages=list(j + 1 for j in range(i + 1)))

      transform.LowerToLLVMOp()


def main():
  problem_size_list = [[60, 60, 60]]

  for sizes in problem_size_list:
    problem = ProblemInstance(EinsumProblem('mk,kn', 'mnk', 2),
                              [np.float32] * 3)

    problem.compile_with_schedule_builder(
        entry_point_name='main',
        fun_to_benchmark_name='matmul',
        compile_time_problem_sizes_dict={k: v for k, v in zip(keys, sizes)},
        schedule_builder=add_tile_schedule,
        dump_ir_to_file='/tmp/abc.mlir')

    problem.run(
        n_iters=100,
        entry_point_name='main',
        runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict,
        dump_obj_to_file='/tmp/abc.o')


if __name__ == '__main__':
  main()
