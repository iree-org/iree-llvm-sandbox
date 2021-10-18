# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .matmul import *

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    SingleTilingExpert(
        sizes=[8, 16, 32],
        interchange=[0, 1, 2],
        peel=False,
        pad=True,
        pack_padding=[0, 1, 2],
        hoist_padding=[2, 3, 0]),
    DoubleTilingExpert(
        sizes1=[256, 128, 256],
        interchange1=[1, 2, 0],
        peel1=False,
        pad1=False,
        pack_padding1=[],
        hoist_padding1=[0],
        sizes2=[8, 16, 32],
        interchange2=[0, 1, 2],
        peel2=False,
        pad2=True,
        pack_padding2=[0, 1],
        hoist_padding2=[3, 4])
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'N', 'K']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  problem_size_list = [[128, 192, 256], [104, 96, 108]]
  for np_types in [[np.float32, np.float32, np.float32]]:
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
            problem_definition=MatmulProblem(),
            problem_sizes_keys=keys,
            np_types=np_types)

        problem.compile(
            entry_point_name='matmul_main',
            fun_to_benchmark_name='matmul_on_tensors',
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert)

        problem.run(
            n_iters=n_iters,
            entry_point_name='matmul_main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict)

      # For single-threaded apples-to-apples comparisons, run with:
      # MKL_NUM_THREADS=1
      import os
      if os.environ.get('BENCHMARK_NUMPY'):
        print('Numpy')
        A, B, C = MatmulProblem().tensors_np_builder(*problem_sizes, *np_types)

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            C.fill(0.)
            np.dot(A, B, out=C)

        timed_invoke(
            run_n_iters,
            MatmulProblem().gflop_count_builder(*problem_sizes),
            n_iters=n_iters)

      # For single-threaded apples-to-apples comparisons, run with:
      # ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
      if os.environ.get('BENCHMARK_TORCH'):
        print('Torch')
        import torch
        torch.set_num_threads(1)
        A, B, C = [
            torch.from_numpy(t) for t in MatmulProblem().tensors_np_builder(
                *problem_sizes, *np_types)
        ]

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            C.fill_(0.)
            torch.mm(A, B, out=C)

        timed_invoke(
            run_n_iters,
            MatmulProblem().gflop_count_builder(*problem_sizes),
            n_iters=n_iters)


if __name__ == '__main__':
  main()
