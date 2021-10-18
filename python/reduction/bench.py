# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'reduction_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

# TODO: activate post core changes.
# all_experts = [
#     SingleTilingExpert(
#         fun_name=fun_name,
#         op_name=op_name,
#         sizes=[16, 4],
#         interchange=[0, 1],
#         peel=[0, 1],
#         pad=False,
#         pack_padding=[],
#         hoist_padding=[],
#         print_ir_after_all=True)
# ]


class LoweringOnly(TransformationList):

  def __init__(self, tiling_transforms):
    t = tiling_transforms + [Bufferize(), LowerVectors(), LowerToLLVM()]
    TransformationList.__init__(self, **{'transforms': t})


# TODO: Check generate code for basic code quality, e.g., no linalg.copy.

# No tiling.
expert_no_tiling = LoweringOnly([])

all_experts = [expert_no_tiling]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'K']


# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  problem_size_list = [[128, 192], [104, 96], [8000, 6000]]
  for np_types in [[np.float32, np.float32]]:
    for problem_sizes in problem_size_list:
      compile_time_problem_sizes_dict = {
          k: v for k, v in zip(keys, problem_sizes)
      }
      runtime_problem_sizes_dict = compile_time_problem_sizes_dict
      # Init printing.
      print(
          f'\n###############################################################\n'
          f'Problem size {compile_time_problem_sizes_dict}\n'
          f'Problem types {np_types}')
      for expert in all_experts:
        problem = ProblemInstance(
            problem_definition=Reduction2dProblem(),
            problem_sizes_keys=keys,
            np_types=np_types)

        problem.compile(
            entry_point_name='main',
            fun_to_benchmark_name=fun_name,
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert)

        problem.run(
            n_iters=n_iters,
            entry_point_name='main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict)

      # For single-threaded apples-to-apples comparisons, run with:
      # MKL_NUM_THREADS=1
      import os
      if os.environ.get('BENCHMARK_NUMPY'):
        print('Numpy')
        A, B = Reduction2dProblem().tensors_np_builder(*problem_sizes,
                                                       *np_types)

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            B.fill(0.)
            np.sum(A, axis=1, out=B)

        timed_invoke(
            run_n_iters,
            Reduction2dProblem().gflop_count_builder(*problem_sizes),
            n_iters=n_iters)

      # For single-threaded apples-to-apples comparisons, run with:
      # ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
      if os.environ.get('BENCHMARK_TORCH'):
        print('Torch')
        import torch
        torch.set_num_threads(1)
        A, B = [
            torch.from_numpy(t)
            for t in Reduction2dProblem().tensors_np_builder(
                *problem_sizes, *np_types)
        ]

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            B.fill_(0.)
            torch.sum(A, dim=1, out=B)

        timed_invoke(
            run_n_iters,
            Reduction2dProblem().gflop_count_builder(*problem_sizes),
            n_iters=n_iters)


if __name__ == '__main__':
  main()
