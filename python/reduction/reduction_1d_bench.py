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


def all_experts(problem_sizes: List[int]):
  return [
      TileAndDecompose(
          fun_name=fun_name,
          op_name=op_name,
          tile_sizes=[32])\
      .then(Vectorize(fun_name, op_name))\
      .then(Bufferize())\
      .then(StagedVectorLowering(multi_reduction_lowering='innerreduction'))\
      .then(LowerToLLVM())\
      .print_ir(after_all=False),
  ]

################################################################################
### Problem instantiations.
################################################################################

keys = ['M']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 100
  problem_size_list = [
      [128],
      [104],
      [256],
      [1000],
      [8000],
  ]

  def numpy_kernel(args, sizes, types):
    A, B = args
    B.fill(0.)
    np.sum(A, out=B)

  def pytorch_kernel(args, sizes, types):
    A, B = args
    B.fill_(0.)
    torch.sum(A, out=B)

  for problem_sizes in problem_size_list:
    test_harness(
        lambda s, t: Reduction1DProblem(), [[np.float32] * 2],
        map(make_size_list, [problem_sizes]),
        all_experts(problem_sizes),
        n_iters=n_iters,
        function_name=fun_name)


if __name__ == '__main__':
  main()
