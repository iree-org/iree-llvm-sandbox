# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from numpy.testing._private.utils import KnownFailureException
from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *
from ..core.transform import *

from .definitions import *

fun_name = 'reduction_1d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################

all_names = [
  "Reduction1dExpert"
]
def all_experts(problem_sizes: List[int]):
  return [
      TileAndDecompose(
          fun_name=fun_name,
          op_name=op_name,
          tile_sizes=[32])\
      .then(Vectorize(fun_name, op_name))\
      .then(Bufferize())\
      .then(LowerVectors(multi_reduction_lowering='innerreduction'))\
      .then(LowerToLLVM())\
      .print_ir(after_all=False),
  ]


################################################################################
### Problem instantiations.
################################################################################

keys = ['M']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
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
    default_dynamic_at_compile_time_list=[],
    default_spec_list=[])

  def numpy_kernel(args, sizes, types):
    A, B = args
    B.fill(0.)
    np.sum(A, out=B)

  def pytorch_kernel(args, sizes, types):
    A, B = args
    B.fill_(0.)
    torch.sum(A, out=B)

  for problem_sizes in args.problem_sizes_list:
    test_harness(lambda s, t: Reduction1DProblem(), [[np.float32] * 2],
          test_sizes(keys, [problem_sizes]),
          test_experts(all_experts(problem_sizes), all_names, args.expert_list),
          n_iters=args.n_iters,
          dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
