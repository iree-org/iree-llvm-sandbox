# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

import typing as tp

fun_name = 'matmul'
op_name = 'linalg.generic'

################################################################################
# Compilation strategies.
################################################################################

all_names = [ \
  'WeakScalingParallelStrategy',
            ]

# Some experimental configs while we bring up parallelism.
num_cores = 8
num_async_threads = 8
num_tiles_per_async_thread = 4
parallel_tile_size_1, parallel_tile_size_2, parallel_tile_size_3 = 96, 128, 255
parallel_tile_size_per_async_thread = parallel_tile_size_1 * num_tiles_per_async_thread

# Throw a wrench in the problem size.
wrench = -11
total_problem_size_1 = \
  num_async_threads * num_tiles_per_async_thread * parallel_tile_size_1 - wrench


def all_experts(fun_name: str):
  return [
    e.print_pipeline(before_all=False) for e in [                             \
      e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [      \
        LinalgExtTile(fun_name,
                      op_name,
                      tile_sizes=[parallel_tile_size_per_async_thread])
          .then(LinalgExtTileToInParallel(fun_name, op_name))
          .then(DoubleTile(fun_name,
                           op_name,
                           tile_sizes1=[parallel_tile_size_1,
                                        parallel_tile_size_2,
                                        parallel_tile_size_3],
                           tile_interchange1=[0, 2, 1],
                           tile_sizes2=[12, 32, 1],
                           tile_interchange2=[1, 0, 2],
                           # In the parallel case, peeling performs quite better atm.
                           # TODO: Investigate inefficiencies in padding/packing.
                           # peel2=[0, 1, 2],
                           pad2=True,
                           pack_paddings2=[1, 1, 0],
                           hoist_paddings2=[1, 2, 0],
                           transpose_paddings2=[[1, 0], [0, 1], [0, 1]],
                           ))
          .then(Vectorize(fun_name, ''))
          .then(Bufferize)
          .then(LinalgExtInParallelToAsync)
          .then(LowerVectors)
          .then(LowerToLLVM)
        ]
    ]
  ]


################################################################################
# Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
      "matmul benchmark",
      default_n_iters=100,
      default_problem_sizes_list=[[total_problem_size_1, 1234, 2345]],
      default_expert_list=all_names,
      default_dynamic_at_compile_time_list=[
          [],  # case 1: static at compile time
      ],
      default_spec_list=[
          'mk,kn',  # C += A.B
      ])

  for dynamic_at_compile_time in args.dynamic_at_compile_time_list:
    for spec in args.spec_list:

      def pytorch_kernel(args, sizes, types):
        import torch
        torch.set_num_threads(num_cores)
        A, B, C = args
        C.fill_(0.)
        if spec == 'km,kn':
          A = np.transpose(A)
        if spec == 'mk,nk':
          B = np.transpose(B)
        torch.mm(A, B, out=C)

      func_with_spec = fun_name + '_' + spec
      func_with_spec = func_with_spec.replace(',', '')

      test_harness(lambda s, t: EinsumProblem(spec, 'mnk', 2),
                   [[np.float32] * 3],
                   test_sizes(keys, args.problem_sizes_list),
                   test_experts(all_experts(func_with_spec), all_names,
                                args.expert_list),
                   n_iters=args.n_iters,
                   dynamic_at_compile_time_sizes=set(
                       dynamic_at_compile_time).intersection(keys),
                   function_name=func_with_spec,
                   dump_ir_to_file='/tmp/abc.mlir',
                   dump_obj_to_file='/tmp/abc.o',
                   dump_data_to_file=args.dump_data,
                   pytorch_benchmark=pytorch_kernel)


if __name__ == '__main__':
  main()
