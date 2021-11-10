# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'conv3d_ndhwc_dhwcf_main'
op_name = 'linalg.conv_3d_ndhwc_dhwcf'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    SingleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        #      N  D, H  W  C   KH KW F
        sizes=[1, 2, 1, 8, 32, 1, 1, 8],
        interchange=[],
        peel=[],
        pad=False,
        pack_paddings=[],
        hoist_paddings=[],
        print_ir_after_all=False)
]

################################################################################
### Problem instantiation
################################################################################

keys = ['N', 'D', 'H', 'W', 'C', 'KD', 'KH', 'KW', 'F', 'strides', 'dilations']


# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  #   N  D   H   W   C  KD  KH  KW   F     st      dil
  problem_size_list = [\
     [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 1, 1], [1, 1, 1]], \
     [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 1, 2], [1, 1, 2]], \
     [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 2, 1], [1, 1, 2]], \
     [8, 4, 16, 16, 32,  3,  3,  3, 64, [1, 2, 2], [1, 2, 2]]  \
  ]
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
            problem_definition=ConvolutionProblem(
                'NDHWC',
                'DHWCF',
                strides=compile_time_problem_sizes_dict['strides'],
                dilations=compile_time_problem_sizes_dict['dilations']),
            problem_sizes_keys=keys,
            np_types=np_types)
        assert problem.problem_definition.keys() == keys

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
