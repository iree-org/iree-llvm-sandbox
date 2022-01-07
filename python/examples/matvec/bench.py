# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

################################################################################
### Compilation strategies.
################################################################################

all_experts = [e.print_ir(after_all=False) for e in [
    SingleTilingExpert(
        'matvec_on_tensors',
        'linalg.generic',
        tile_sizes=[12, 32],
        tile_interchange=[0, 1],
        pad=True,
        pack_paddings=[1, 1, 0],
        hoist_paddings=[2, 3, 0]),
    DoubleTilingExpert(
        'matvec_on_tensors',
        'linalg.generic',
        tile_sizes1=[128, 128],
        tile_interchange1=[0, 1],
        tile_sizes2=[12, 32],
        tile_interchange2=[0, 1],
        pad2=True,
        pack_paddings2=[1, 1, 0],
        hoist_paddings2=[4, 3, 0])
]]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 100

  # Specify default configuration and parse command line.
  args = test_argparser(
    "matvec benchmark",
    default_problem_sizes_list = [
      [192, 128],
      [260, 280],
      [1000, 1000],
      [1024, 1024],
      [2040, 2040],
      [4000, 4000],
    ],
    default_expert_list = [
      idx for idx, _ in enumerate(all_experts)
    ],
    default_dynamic_at_compile_time_list = [],
    default_spec_list = [])

  def numpy_kernel(args, sizes, types):
    A, y, x = args
    x.fill(0.)
    np.dot(A, y, out=x)

  def pytorch_kernel(args, sizes, types):
    A, y, x = args
    x.fill_(0.)
    torch.mv(A, y, out=x)

  test_harness(
      lambda s, t: EinsumProblem('mn,n'), [[np.float32] * 3],
      map(make_size_list, args.problem_sizes_list),
      [all_experts[idx] for idx in args.expert_list if idx < len(all_experts)],
      n_iters=n_iters,
      function_name='matvec_on_tensors',
      numpy_benchmark=numpy_kernel,
      pytorch_benchmark=pytorch_kernel)

if __name__ == '__main__':
  main()
