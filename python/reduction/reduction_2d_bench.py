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


class ExperimentalSplitAndFuseFillOpExpert(TransformationList):

  def __init__(self, transforms, **kwargs):
    t = transforms + [
        # TODO: bufferization before vectorization triggers
        # a bufferization issue atm.
        # This is inefficient wrt VTW/VTR forwarding.
        #Bufferize()
    ] + StagedLowerVectorsTransformationList() + [LowerToLLVM()]
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


experimental_tile_and_fuse_expert = \
  ExperimentalSplitAndFuseFillOpExpert( \
    [ \
      ExperimentalSplitAndFuseFillOp( \
        fun_name=fun_name, op_name=op_name, tile_sizes=[4, 4]), \
      Bufferize(), \
      Vectorize(fun_name=fun_name, op_name=op_name) \
    ], \
  print_ir_after_all=False)


def all_experts(problem_sizes: List[int]):
  return [
      SingleTilingExpert(
          fun_name=fun_name,
          op_name=op_name,
          # Little trick avoids tiling small dimensions and otherwise tile by 128.
          sizes=[4, 128] if problem_sizes[1] > 256 else [4],
          interchange=[],
          peel=[],
          pad=False,
          pack_paddings=[],
          hoist_paddings=[],
          # kwargs passed down to LowerVectors.
          # TODO: better composition of experts.
          multi_reduction_lowering='innerreduction',
          print_ir_after_all=False),
      # experimental_tile_and_fuse_expert
  ]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'K']


# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  problem_size_list = [
      [128, 256],
      [104, 128],
      [256, 256],
      [1000, 1024],
      [8000, 6144],
  ]
  for np_types in [[np.float32, np.float32]]:
    for problem_sizes in problem_size_list:
      compile_time_problem_sizes_dict = {
          k: v for k, v in zip(keys, problem_sizes)
      }
      runtime_problem_sizes_dict = compile_time_problem_sizes_dict
      # Init printing.
      print(f'\n#############################################################\n'
            f'Compile-time problem sizes {compile_time_problem_sizes_dict}\n'
            f'Runtime problem sizes {runtime_problem_sizes_dict}\n'
            f'Problem types {np_types}')

      for expert in all_experts(problem_sizes):
        print(f'\nCompilation expert {expert}')

        problem = ProblemInstance(
            problem_definition=Reduction2DProblem(),
            problem_sizes_keys=keys,
            np_types=np_types)

        problem.compile(
            entry_point_name='main',
            fun_to_benchmark_name=fun_name,
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert,
            # Used to pipe through llvm-mca
            # dump_ir_to_file='/tmp/abc.mlir'
        )

        problem.run(
            n_iters=n_iters,
            entry_point_name='main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict)

      # For single-threaded apples-to-apples comparisons, run with:
      # MKL_NUM_THREADS=1
      import os
      if os.environ.get('BENCHMARK_NUMPY'):
        print('Numpy')
        A, B = Reduction2DProblem().tensors_np_builder(*problem_sizes,
                                                       *np_types)

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            B.fill(0.)
            np.sum(A, axis=1, out=B)

        timed_invoke(
            run_n_iters,
            Reduction2DProblem().gflop_count_builder(*problem_sizes),
            n_iters=n_iters)

      # For single-threaded apples-to-apples comparisons, run with:
      # ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
      if os.environ.get('BENCHMARK_TORCH'):
        print('Torch')
        import torch
        torch.set_num_threads(1)
        A, B = [
            torch.from_numpy(t)
            for t in Reduction2DProblem().tensors_np_builder(
                *problem_sizes, *np_types)
        ]

        def run_n_iters(n_iters: int):
          for _ in range(n_iters):
            B.fill_(0.)
            torch.sum(A, dim=1, out=B)

        timed_invoke(
            run_n_iters,
            Reduction2DProblem().gflop_count_builder(*problem_sizes),
            n_iters=n_iters)


if __name__ == '__main__':
  main()
