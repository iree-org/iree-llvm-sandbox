# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *
from .ops import *

fun_name = 'transpose_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    DoubleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        sizes1=[256, 256],
        interchange1=[],
        peel1=[],
        pad1=False,
        pack_paddings1=[],
        hoist_paddings1=[],
        sizes2=[8, 8],
        interchange2=[],
        peel2=[],
        pad2=True,
        pack_paddings2=[0, 1],
        hoist_paddings2=[2, 2],
        # kwargs passed down to LowerVectors.
        # TODO: better composition of experts.
        transpose_lowering='shuffle',
        # Set to True to see the IR.
        print_ir_after_all=False),
    TripleTilingExpert(
        fun_name=fun_name,
        op_name=op_name,
        sizes1=[256, 256],
        interchange1=[],
        peel1=[],
        pad1=False,
        pack_paddings1=[],
        hoist_paddings1=[],
        sizes2=[32, 32],
        interchange2=[],
        peel2=[],
        pad2=False,
        pack_paddings2=[],
        hoist_paddings2=[],
        sizes3=[8, 8],
        interchange3=[],
        peel3=[],
        pad3=True,
        pack_paddings3=[0, 1],
        hoist_paddings3=[3, 3],
        # kwargs passed down to LowerVectors.
        # TODO: better composition of experts.
        transpose_lowering='shuffle',
        # Set to True to see the IR.
        print_ir_after_all=False)
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M', 'N']


# CHECK-NOT: FAILURE
def main():
  n_iters = 10
  problem_size_list = [
      # Too small to be really interesting.
      [128, 192],
      [104, 96],
      [240, 256],
      [528, 512],
      [1056, 1024],

      # TODO: this is too slow atm.
      [4096, 4096],
      [6912, 4608],
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
            problem_definition=TransposeNDProblem(
                permutation=[1, 0], op_builder=transpose_2d),
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
            runtime_problem_sizes_dict=runtime_problem_sizes_dict,
            # Used to pipe through llvm-mca with the **actual JIT'ed object**.
            # dump_obj_to_file='/tmp/abc.o'
        )


if __name__ == '__main__':
  main()
