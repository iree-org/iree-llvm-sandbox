# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *

fun_name = 'transpose_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    DoubleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        sizes1=[32, 32],
        interchange1=[],
        peel1=False,
        pad1=False,
        pack_padding1=[],
        hoist_padding1=[],
        sizes2=[8, 8],
        interchange2=[],
        peel2=False,
        pad2=False,
        pack_padding2=[],
        hoist_padding2=[],
        # Set to True to see the IR.
        print_ir_after_all=False)
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'N']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  problem_size_list = [
      # Too small to be interesting.
      # [128, 192],
      # [104, 96],
      [256, 256],
      [512, 512],
      [1024, 1024],

      # TODO: this is too slow atm.
      # [8192, 6144],
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

      for expert in all_experts:
        print(f'\nCompilation expert {expert}')

        problem = ProblemInstance(
            problem_definition=Transpose2dProblem(),
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


if __name__ == '__main__':
  main()
