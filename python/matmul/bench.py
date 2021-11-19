# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    SingleTilingExpert(
        'matmul_on_tensors',
        'linalg.generic',
        sizes=[12, 32, 8],
        interchange=[0, 1, 2],
        peel=[],
        pad=True,
        pack_paddings=[1, 1, 0],
        hoist_paddings=[2, 3, 0],
        print_ir_after_all=False),
    DoubleTilingExpert(
        'matmul_on_tensors',
        'linalg.generic',
        sizes1=[288, 128, 512],
        interchange1=[0, 2, 1],
        peel1=[],
        pad1=False,
        pack_paddings1=[],
        hoist_paddings1=[],
        sizes2=[9, 32, 16],
        interchange2=[0, 1, 2],
        peel2=[],
        pad2=True,
        pack_paddings2=[1, 1, 0],
        hoist_paddings2=[5, 6, 0],
        # kwargs start here
        # kwargs passed down to LowerVectors.
        # TODO: better composition of experts.
        transpose_lowering='eltwise',
        # Set to True to see the IR.
        print_ir_after_all=False)
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'N', 'K']


# CHECK-NOT: FAILURE
def main():
  n_iters = 10
  problem_size_list = [
      [192, 128, 256],
      [260, 280, 300],
      [1000, 1000, 1000],
      [1024, 1024, 1024],
      [2040, 2040, 2040],
      [2040, 2041, 2042],
      [2048, 2048, 2048],
      [4000, 4000, 4000],
  ]
  for np_types in [[np.float32, np.float32, np.float32]]:
    for problem_sizes in problem_size_list:
      runtime_problem_sizes_dict = {k: v for k, v in zip(keys, problem_sizes)}
      for compile_time_problem_sizes_dict in [                      \
          # case 1: static at compile time

          runtime_problem_sizes_dict,                               \
          {                                                         \
           # case 2: partially dynamic at compile time

            k: v for k, v in zip(keys, [-1, problem_sizes[1], -1])  \
          },                                                        \
          {                                                         \
           # case 3: fully dynamic at compile time

            k: v for k, v in zip(keys, [-1, -1, -1])                \
          }]:
        # Init printing.
        print(
            f'\n###############################################################\n'
            f'Runtime problem size {runtime_problem_sizes_dict}\n'
            f'Compile-time problem size {compile_time_problem_sizes_dict}\n'
            f'Problem types {np_types}')
        for expert in all_experts:
          problem = ProblemInstance(
              problem_definition=EinsumProblem('mk,kn'),
              problem_sizes_keys=keys,
              np_types=np_types)

          problem.compile(
              entry_point_name='matmul_main',
              fun_to_benchmark_name='matmul_on_tensors',
              compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
              transform=expert,
              # Used to pipe through llvm-mca
              dump_ir_to_file='/tmp/abc.mlir')

          problem.run(
              n_iters=n_iters,
              entry_point_name='matmul_main',
              runtime_problem_sizes_dict=runtime_problem_sizes_dict,
              # Used to pipe through llvm-mca with the **actual JIT'ed object**.
              dump_obj_to_file='/tmp/abc.o')

        # For single-threaded apples-to-apples comparisons, run with:
        # MKL_NUM_THREADS=1
        import os
        if os.environ.get('BENCHMARK_NUMPY'):
          print('Numpy')
          A, B, C = EinsumProblem('mk,kn').tensors_np_builder(
              *problem_sizes, *np_types)

          def run_n_iters(n_iters: int):
            for _ in range(n_iters):
              C.fill(0.)
              np.dot(A, B, out=C)

          timed_invoke(
              run_n_iters,
              EinsumProblem('mk,kn').gflop_count_builder(*problem_sizes),
              EinsumProblem('mk,kn').gbyte_count_builder(*problem_sizes),
              n_iters=n_iters)

        # For single-threaded apples-to-apples comparisons, run with:
        # ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
        if os.environ.get('BENCHMARK_TORCH'):
          print('Torch')
          import torch
          torch.set_num_threads(1)
          A, B, C = [
              torch.from_numpy(t)
              for t in EinsumProblem('mk,kn').tensors_np_builder(
                  *problem_sizes, *np_types)
          ]

          def run_n_iters(n_iters: int):
            for _ in range(n_iters):
              C.fill_(0.)
              torch.mm(A, B, out=C)

          timed_invoke(
              run_n_iters,
              EinsumProblem('mk,kn').gflop_count_builder(*problem_sizes),
              EinsumProblem('mk,kn').gbyte_count_builder(*problem_sizes),
              n_iters=n_iters)


if __name__ == '__main__':
  main()
