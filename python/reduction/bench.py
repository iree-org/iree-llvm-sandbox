# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .reduction import *

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    SingleTilingExpert(
        sizes=[8, 128],
        interchange=[0, 1],
        peel=[0, 1],
        pad=False,
        pack_padding=[],
        hoist_padding=[])
]

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
            problem_sizes_keys=keys,
            np_types=np_types,
            shapes_from_list_of_sizes_builder=reduction_2d_shapes_builder,
            compile_time_function_types_mlir_builder=reduction_2d_types_mlir_builder,
            fun_to_compile_mlir_builder=build_reduction_2d_under_context_manager
        )

        problem.compile(
            entry_point_name='reduction_2d_main',
            fun_to_benchmark_name='reduction_2d_on_tensors',
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert)

        problem.run(
            n_iters=n_iters,
            entry_point_name='reduction_2d_main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict,
            runtime_data_np_builder=reduction_2d_tensors_np_builder,
            gflop_count_builder=reduction_2d_gflop_count_builder,
            check_fun=reduction_2d_check)

      # For single-threaded apples-to-apples comparisons, run with:
      # MKL_NUM_THREADS=1
      import os
      if True or os.environ.get('BENCHMARK_NUMPY'):
        print('Numpy')
        A, B = reduction_2d_tensors_np_builder(*problem_sizes, np_types)

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            B.fill(0.)
            np.sum(A, axis=1, out=B)

        timed_invoke(
            run_n_iters,
            reduction_2d_gflop_count_builder(*problem_sizes),
            n_iters=n_iters)

      # For single-threaded apples-to-apples comparisons, run with:
      # ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
      if os.environ.get('BENCHMARK_TORCH'):
        print('Torch')
        import torch
        torch.set_num_threads(1)
        A, B = [
            torch.from_numpy(t)
            for t in reduction_2d_tensors_np_builder(*problem_sizes, np_types)
        ]

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            B.fill_(0.)
            torch.sum(A, dim=1, out=B)

        timed_invoke(
            run_n_iters,
            reduction_2d_gflop_count_builder(*problem_sizes),
            n_iters=n_iters)


if __name__ == '__main__':
  main()
