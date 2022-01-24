# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

fun_name = 'row_reduction_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

# Note: `\` char at the end of next line prevents formatter reflows, keep it.
all_names = [ \
  "RowReduction2DExpert" \
  ]


def all_experts(problem_sizes: List[int]):
  return [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [ \
      TileAndDecompose(
          fun_name=fun_name,
          op_name=op_name,
          # Little trick avoids tiling small dimensions and otherwise tile by 128.
          tile_sizes=[4, 128] if problem_sizes[1] > 256 else [4])
        .then(Vectorize(fun_name, op_name))
        .then(LoweringOnlyExpert(fun_name,
                                 op_name,
                                 multi_reduction_lowering='innerreduction')),
    ]
  ]


################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  # Note: `\` char at the end of next line prevents formatter reflows, keep it.
  args = test_argparser(  \
    "row reduction 2d benchmark",
    default_n_iters=100,
    default_problem_sizes_list=[
      [128, 256],
      [104, 128],
      [256, 256],
      [1000, 1024],
      [8000, 6144],
    ],
    default_expert_list=all_names,
    default_dynamic_at_compile_time_list=[],
    default_spec_list=[])

  def numpy_kernel(args, sizes, types):
    A, B = args
    B.fill(0.)
    np.sum(A, axis=1, out=B)

  def pytorch_kernel(args, sizes, types):
    A, B = args
    B.fill_(0.)
    torch.sum(A, dim=1, out=B)

  for problem_sizes in args.problem_sizes_list:
    test_harness(lambda s, t: EinsumProblem('mn->m', 'mn', 1), [[np.float32] * 2],
                 test_sizes(keys, [problem_sizes]),
                 test_experts(all_experts(problem_sizes), all_names,
                              args.expert_list),
                 n_iters=args.n_iters,
                 function_name=fun_name,
                 dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
