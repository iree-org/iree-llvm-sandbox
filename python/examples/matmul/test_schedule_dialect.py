# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

import mlir.iree_sandbox as sandbox
import mlir.ir as ir
import mlir.dialects.pdl as pdl
import mlir.dialects.linalg_transform as transform

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']

# CHECK-NOT: FAILURE


def add_tile_schedule(module):
  with InsertionPoint(module.body):
    pdl_pattern_name = 'pdl_pattern_name'
    pdl_pattern = pdl.PatternOp(benefit=1, name=pdl_pattern_name)
    with ir.InsertionPoint(pdl_pattern.body):
      args = pdl.OperandsOp()
      types = pdl.TypesOp()
      pdl_op = pdl.OperationOp('linalg.generic', args=[args], types=[types])
      pdl.RewriteOp(pdl_op, 'linalg_transform.apply')

    sequence = transform.SequenceOp()
    with ir.InsertionPoint(sequence.body.blocks[0]):
      transform.TileOp(pdl_pattern_name, sizes=[12, 32, 1], pad=False)
      transform.BufferizeOp()
      transform.LowerVectorsOp()
      transform.LowerToLLVMOp()


def main():
  problem_size_list = [[1000, 1000, 1000]]

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
        n_iters=1,
        entry_point_name='main',
        runtime_problem_sizes_dict=problem.compile_time_problem_sizes_dict,
        dump_obj_to_file='/tmp/abc.o')


if __name__ == '__main__':
  main()
