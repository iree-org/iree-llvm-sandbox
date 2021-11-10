# RUN: %PYTHON %s 2>&1 | FileCheck %s

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from .definitions import *
from .ops import *

fun_name = 'transpose_4d_on_tensors'
op_name = 'linalg.generic'

expert_transpose_4d_0213 = SingleTilingExpert(
    fun_name=fun_name,
    op_name=op_name,
    sizes=[1, 4, 4, 16],
    interchange=[],
    peel=[0, 1],
    pad=False,
    pack_paddings=[],
    hoist_paddings=[],
    transpose_lowering='shuffle',
    # kwargs start here.
    # transforms=[Print('Input IR')],
    print_ir_after_all=False)

expert_transpose_4d_1302 = SingleTilingExpert(
    fun_name=fun_name,
    op_name=op_name,
    sizes=[1, 0, 4, 4],
    interchange=[],
    peel=[0, 1],
    pad=False,
    pack_paddings=[],
    hoist_paddings=[],
    transpose_lowering='shuffle',
    # kwargs start here.
    # transforms=[Print('Input IR')],
    print_ir_after_all=False)

################################################################################
### Problem instantiations.
################################################################################


# CHECK-NOT: FAILURE
def main():
  n_iters = 25
  keys = ['M', 'N', 'K', 'L']

  # List of problems (sizes, permutation, op, expert strategy).
  problem_list = [
      # [16, 100, 8, 128] -> [16, 8, 100, 128]
      ([16, 100, 8, 128], [0, 2, 1,
                           3], transpose_4d_0213, expert_transpose_4d_0213),
      # [64, 224, 224, 3] -> [224, 3, 64, 224]
      ([64, 224, 224, 3], [1, 3, 0,
                           2], transpose_4d_1302, expert_transpose_4d_1302),
      # [128, 224, 224, 3] -> [224, 3, 224, 128]
      ([128, 224, 224, 3], [1, 3, 0,
                            2], transpose_4d_1302, expert_transpose_4d_1302),
      # [8, 224, 224, 3] -> [224, 3, 8, 224]
      ([8, 224, 224, 3], [1, 3, 0,
                          2], transpose_4d_1302, expert_transpose_4d_1302),
  ]

  for problem in problem_list:
    compile_time_problem_sizes_dict = {k: v for k, v in zip(keys, problem[0])}
    runtime_problem_sizes_dict = compile_time_problem_sizes_dict
    # Init printing.
    print(f'\n#############################################################\n'
          f'Compile-time problem sizes {compile_time_problem_sizes_dict}\n'
          f'Runtime problem sizes {runtime_problem_sizes_dict}')

    expert = problem[3]
    print(f'Compilation expert {expert}')

    instance = ProblemInstance(
        problem_definition=TransposeNDProblem(
            permutation=problem[1], op_builder=problem[2]),
        problem_sizes_keys=keys,
        np_types=[np.float32, np.float32])

    instance.compile(
        entry_point_name='main',
        fun_to_benchmark_name='transpose_4d_on_tensors',
        compile_time_problem_sizes_dict=compile_time_problem_sizes_dict,
        transform=expert)

    instance.run(
        n_iters=n_iters,
        entry_point_name='main',
        runtime_problem_sizes_dict=runtime_problem_sizes_dict)


if __name__ == '__main__':
  main()
