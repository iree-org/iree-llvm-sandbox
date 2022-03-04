# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from numpy.testing._private.utils import KnownFailureException
from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *
from ..core.transform import *

from ..contraction.definitions import *

fun_name = 'reduction_1d'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

# Note: `\` char at the end of next line prevents formatter reflows, keep it.
all_names = [         \
  "Tile1DPeel"
            ]


def all_experts(problem_sizes: List[int]):
  return [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [ \
      Tile(fun_name=fun_name,
           op_name=op_name,
           tile_sizes=[512],
           peel=[0])
      .then(Vectorize(fun_name, ''))
      .then(LoweringOnlyExpert(fun_name,
                                op_name,
                                multi_reduction_lowering='innerreduction')),
    ]
  ]


################################################################################
### Problem instantiations.
################################################################################

keys = ['m']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  # Note: `\` char at the end of next line prevents formatter reflows, keep it.
  args = test_argparser(  \
    "reduction 1d benchmark",
    default_n_iters=100,
    default_problem_sizes_list=[
      [128],
      [104],
      [256],
      [1000],
      [8000],
    ],
    default_expert_list=all_names,
    default_dynamic_at_compile_time_list=[
      [],
      ['m']
    ],
    default_spec_list=[])

  def numpy_kernel(args, sizes, types):
    A, B = args
    B.fill(0.)
    np.sum(A, out=B)

  def pytorch_kernel(args, sizes, types):
    A, B = args
    B.fill_(0.)
    torch.sum(A, out=B)

  for dynamic_at_compile_time in args.dynamic_at_compile_time_list:
    for problem_sizes in args.problem_sizes_list:
      test_harness(lambda s, t: EinsumProblem('m->', 'm', 1),
                   [[np.float32] * 2],
                   test_sizes(keys, [problem_sizes]),
                   test_experts(all_experts(problem_sizes), all_names,
                                args.expert_list),
                   n_iters=args.n_iters,
                   dynamic_at_compile_time_sizes=set(
                       dynamic_at_compile_time).intersection(keys),
                   function_name=fun_name,
                   dump_ir_to_file='/tmp/abcd.mlir',
                   dump_obj_to_file='/tmp/abcd.o',
                   dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
