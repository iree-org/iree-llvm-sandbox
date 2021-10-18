# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .reduction import *

################################################################################
### Compilation strategies.
################################################################################


class TestExpert(TransformationList):

  def __init__(self, tiling_transforms):
    t = tiling_transforms + [Bufferize(), LowerVectors(), LowerToLLVM()]
    TransformationList.__init__(self, **{'transforms': t})


# TODO: Check generate code for basic code quality, e.g., no linalg.copy.

# No tiling.
expert_no_tiling = TestExpert([])

all_experts = [expert_no_tiling]


################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'K']

# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[24, 32], [27, 37]]
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


if __name__ == '__main__':
  main()
