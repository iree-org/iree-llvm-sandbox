# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.problem_definition import *
from ..core.transforms import *

from .definitions import *

################################################################################
### Compilation strategies.
################################################################################


class LoweringOnly(TransformationList):

  def __init__(self, tiling_transforms, **kwargs):
    t = tiling_transforms + [Bufferize(), LowerVectors(), LowerToLLVM()]
    d = {'transforms': t}
    kwargs.update(d)
    TransformationList.__init__(self, **kwargs)


# TODO: Check generate code for basic code quality, e.g., no linalg.copy.

# No tiling.
expert_no_tiling = LoweringOnly([], print_ir_after_all=False)

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
            problem_definition=Reduction2dProblem(),
            problem_sizes_keys=keys,
            np_types=np_types)

        problem.compile(
            entry_point_name='reduction_2d_main',
            fun_to_benchmark_name='reduction_2d_on_tensors',
            compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
            transform=expert)

        problem.run(
            n_iters=n_iters,
            entry_point_name='reduction_2d_main',
            runtime_problem_sizes_dict=runtime_problem_sizes_dict)


if __name__ == '__main__':
  main()
