# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

################################################################################
### Compilation strategies.
################################################################################

all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [
    SingleTilingExpert('matmul_on_tensors',
                       'linalg.generic',
                       tile_sizes=[12, 32, 8],
                       tile_interchange=[0, 1, 2],
                       pad=True,
                       pack_paddings=[1, 1, 0],
                       hoist_paddings=[2, 3, 0]),
    DoubleTileAndDecompose('matmul_on_tensors',
                            'linalg.generic',
                            tile_sizes1=[288, 128, 512],
                            tile_interchange1=[0, 2, 1],
                            tile_sizes2=[9, 32, 16],
                            tile_interchange2=[0, 1, 2],
                            pad2=True,
                            pack_paddings2=[1, 1, 0],
                            hoist_paddings2=[5, 6, 0]).then(\
      Vectorize('matmul_on_tensors',
                'linalg.generic',
                transpose_lowering='eltwise')).then(\
      LoweringOnlyExpert('matmul_on_tensors',
                         'linalg.generic'))
    ]]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}

# CHECK-NOT: FAILURE
def main():
  n_iters = 1000
  problem_size_list = [
      [192, 128, 256],
      [260, 280, 300],
      [1000, 1000, 1000],
      [1024, 1024, 1024],
      [2040, 2040, 2040],
      [2040, 2041, 2042],
      [2048, 2048, 2048],
      [4000, 4000, 4000],
  ]

  def numpy_kernel(args, sizes, types):
    A, B, C = args
    C.fill(0.)
    np.dot(A, B, out=C)

  def pytorch_kernel(args, sizes, types):
    A, B, C = args
    C.fill_(0.)
    torch.mm(A, B, out=C)

  for runtime_only in [
      [],  # case 1: static at compile time
      ['m', 'k'],  # case 2: partially dynamic at compile time
      keys
  ]:  # case 3: fully dynamic at compile time
    test_harness(lambda s, t: EinsumProblem('mk,kn'), [[np.float32] * 3],
                 map(make_size_list, problem_size_list),
                 all_experts,
                 n_iters=n_iters,
                 runtime_only_sizes=set(runtime_only),
                 function_name='matmul_on_tensors',
                 dump_ir_to_file='/tmp/abc.mlir',
                 dump_obj_to_file='/tmp/abc.o',
                 numpy_benchmark=numpy_kernel,
                 pytorch_benchmark=pytorch_kernel)


if __name__ == '__main__':
  main()
