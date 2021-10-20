# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'conv1d_nwc_wcf_main'
op_name = 'linalg.conv_1d_nwc_wcf'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    SingleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        sizes=[4, 4, 16, 1, 1, 8],
        interchange=[],
        peel=[],
        pad=False,
        pack_padding=[],
        hoist_padding=[],
        print_ir_after_all=False)
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'W', 'C', 'KW', 'F', 'dilation', 'stride']


# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  problem_size_list = [[8, 16, 32, 3, 64, 1, 1], [8, 16, 32, 3, 64, 1, 2],
                       [8, 16, 32, 3, 64, 2, 1], [8, 16, 32, 3, 64, 2, 2]]
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
            problem_definition=Conv1d_NWC_WCF_Problem(
                dilation=compile_time_problem_sizes_dict['dilation'],
                stride=compile_time_problem_sizes_dict['stride']),
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


if __name__ == '__main__':
  main()
