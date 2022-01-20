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

all_names = [                     \
  "SingleTiling2DPeel",          \
  "SingleTiling3DPeel",          \
  "SingleTiling2DPad",               \
  "SingleTiling3DPad",               \
  "DoubleTile2DPadAndHoist",     \
  "DoubleTile3DPadAndHoist",     \
  "DoubleTile2DPadAndHoistLarge" \
]

all_experts = [
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [ \
        SingleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes=[6, 32, 1],
                           tile_interchange=[0, 1, 2],
                           peel=[0, 1, 2]),
        SingleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes=[12, 32, 16],
                           tile_interchange=[0, 1, 2],
                           peel=[0, 1, 2]),
        SingleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes=[12, 32, 1],
                           tile_interchange=[0, 1, 2],
                           pad=True,
                           pack_paddings=[1, 1, 0],
                           hoist_paddings=[2, 3, 0]),
        SingleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes=[12, 32, 16],
                           tile_interchange=[0, 1, 2],
                           pad=True,
                           pack_paddings=[1, 1, 0],
                           hoist_paddings=[2, 3, 0]),
        DoubleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes1=[288, 128, 512],
                           tile_interchange1=[0, 2, 1],
                           tile_sizes2=[12, 32, 1],
                           tile_interchange2=[0, 1, 2],
                           pad2=True,
                           pack_paddings2=[1, 1, 0],
                           hoist_paddings2=[5, 6, 0])
          .then(Vectorize('matmul_on_tensors', 'linalg.generic'))
          .then(UnrollOneParentLoop('matmul_on_tensors',
                                    'vector.contract',
                                    parent_loop_num=1,
                                    unroll_factor=4))
          .then(LoweringOnlyExpert('matmul_on_tensors',
                                   'linalg.generic',
                                   transpose_lowering='eltwise')),
        DoubleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes1=[288, 128, 512],
                           tile_interchange1=[0, 2, 1],
                           tile_sizes2=[12, 32, 16],
                           tile_interchange2=[0, 1, 2],
                           pad2=True,
                           pack_paddings2=[1, 1, 0],
                           hoist_paddings2=[5, 6, 0])
          .then(Vectorize('matmul_on_tensors', 'linalg.generic'))
          .then(LoweringOnlyExpert('matmul_on_tensors',
                                   'linalg.generic',
                                   transpose_lowering='eltwise')),
        DoubleTilingExpert('matmul_on_tensors',
                           'linalg.generic',
                           tile_sizes1=[128, 384, 512],
                           tile_interchange1=[0, 1, 2],
                           tile_sizes2=[12, 32, 1],
                           tile_interchange2=[1, 0, 2],
                           pad2=True,
                           pack_paddings2=[1, 1, 0],
                           hoist_paddings2=[3, 2, 0])
          .then(Vectorize('matmul_on_tensors', 'linalg.generic'))
          .then(LoweringOnlyExpert('matmul_on_tensors',
                                   'linalg.generic',
                                   transpose_lowering='eltwise')),
    ]
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
      "matmul benchmark",
      default_n_iters=100,
      default_problem_sizes_list=[ \
        [1, 384, 384],
        [128, 384, 384],
        [128, 1536, 384],
        [128, 384, 1536],
        [192, 128, 256],
        [260, 280, 300],
        [1000, 1000, 1000],
        [1020, 1020, 1020],
        [1020, 1021, 1022],
        [1024, 1024, 1024],
        [2048, 2048, 347]],
      default_expert_list=all_names,
      default_dynamic_at_compile_time_list=[
          [],  # case 1: static at compile time
          ['m', 'k'],  # case 2: partially dynamic at compile time
          keys  # case 3: fully dynamic at compile time
      ],
      default_spec_list=[
          'km,kn',  # C += A^T.B  fastest
          'mk,kn',  # C += A.B
          'mk,nk'  # C += A.B^T  slowest
      ])

  for dynamic_at_compile_time in args.dynamic_at_compile_time_list:
    for spec in args.spec_list:

      def numpy_kernel(args, sizes, types):
        A, B, C = args
        C.fill(0.)
        if spec == 'km,kn':
          A = np.transpose(A)
        if spec == 'mk,nk':
          B = np.transpose(B)
        np.dot(A, B, out=C)

      def pytorch_kernel(args, sizes, types):
        import torch
        A, B, C = args
        C.fill_(0.)
        if spec == 'km,kn':
          A = np.transpose(A)
        if spec == 'mk,nk':
          B = np.transpose(B)
        torch.mm(A, B, out=C)

      test_harness(lambda s, t: EinsumProblem(spec, 2), [[np.float32] * 3],
                   test_sizes(keys, args.problem_sizes_list),
                   test_experts(all_experts, all_names, args.expert_list),
                   n_iters=args.n_iters,
                   dynamic_at_compile_time_sizes=set(
                       dynamic_at_compile_time).intersection(keys),
                   function_name='matmul_on_tensors',
                   dump_ir_to_file='/tmp/abc.mlir',
                   dump_obj_to_file='/tmp/abc.o',
                   dump_data_to_file=args.dump_data,
                   numpy_benchmark=numpy_kernel,
                   pytorch_benchmark=pytorch_kernel)


if __name__ == '__main__':
  main()
