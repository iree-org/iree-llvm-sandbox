# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

fun_name = 'column_reduction_2d'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

# Note: `\` char at the end of next line prevents formatter reflows, keep it.
all_names = [  \
  "Tile4x8", \
  "Tile4x8FusedOutput", \
  "Tile8x8", \
  "Tile8x8FusedOutput", \
  "Tile16x8", \
  "Tile16x8FusedOutput", \
  "Tile4x64", \
  "Tile4x64FusedOutput", \
  "Tile8x64", \
  "Tile8x64FusedOutput", \
  "Tile16x64", \
  "Tile16x64FusedOutput", \
]


def all_experts(problem_sizes: List[int]):
  tile_sizes = [
    [4, 8], [8, 8], [16, 8], \
    [4, 64], [8, 64], [16, 64], \
  ]

  res = []
  for ts in tile_sizes:
    res.append(
      Tile(fun_name=fun_name, op_name=op_name, tile_sizes=[ts[0], ts[1]])
        .then(Vectorize(fun_name, op_name))
        .then(LoweringOnlyExpert(fun_name,
                                 op_name,
                                 multi_reduction_lowering='innerparallel')),
    )
    res.append(
      Tile(fun_name=fun_name, op_name=op_name, tile_sizes=[ts[0], ts[1]])
        .then(ExperimentalFuseFillIntoTiledReductionOutput(fun_name, op_name))
        .then(Vectorize(fun_name, op_name))
        .then(Vectorize(fun_name, 'linalg.fill', vectorize_only_tiled=True))
        .then(LoweringOnlyExpert(fun_name,
                                 op_name,
                                 multi_reduction_lowering='innerparallel')),
    )
  return [e.print_ir(after_all=False, at_begin=False, llvm=False) for e in res]


################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  # Note: `\` char at the end of next line prevents formatter reflows, keep it.
  args = test_argparser(  \
    "column reduction 2d fused fill benchmark",
    default_n_iters=100,
    default_problem_sizes_list=[
      [128, 256],
      [112, 128],
      [256, 256],
      [1008, 1024],
      [8096, 6144],
    ],
    default_expert_list=all_names,
    default_dynamic_at_compile_time_list=[[]],
    default_spec_list=[])

  def numpy_kernel(args, sizes, types):
    A, B = args
    B.fill(0.)
    np.sum(A, axis=0, out=B)

  def pytorch_kernel(args, sizes, types):
    A, B = args
    B.fill_(0.)
    torch.sum(A, dim=0, out=B)

  for problem_sizes in args.problem_sizes_list:
    test_harness(lambda s, t: EinsumProblem('mn->n', 'mn', 1),
                 [[np.float32] * 2],
                 test_sizes(keys, [problem_sizes]),
                 test_experts(all_experts(problem_sizes), all_names,
                              args.expert_list),
                 n_iters=args.n_iters,
                 dynamic_at_compile_time_sizes=[],
                 function_name=fun_name,
                 zero_at_each_iteration=True,
                 dump_ir_to_file='/tmp/abcd.mlir',
                 dump_obj_to_file='/tmp/abcd.o',
                 dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
