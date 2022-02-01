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

num_threads = 32
num_tiles_per_thread = 16
parallel_tile_size = 288


def all_experts(fun_name: str):
  return [
    e.print_pipeline(before_all=False) for e in [                             \
      e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [      \
        LinalgExtTile(fun_name,
                      op_name,
                      tile_sizes=[parallel_tile_size * num_tiles_per_thread])
          .then(LinalgExtTileToInParallel(fun_name, op_name))
          .then(DoubleTile(fun_name,
                           op_name,
                           tile_sizes1=[parallel_tile_size, 128, 512],
                           tile_interchange1=[0, 2, 1],
                           tile_sizes2=[12, 32, 1],
                           tile_interchange2=[0, 1, 2],
                           # TODO: atm hoisting of allocs doesn't care about
                           # parallel scopes, fix this.
                           peel2=[0, 1, 2],
                           pad2=False,
                           pack_paddings2=[1, 1, 0],
                           hoist_paddings2=[5, 6, 0],
                           transpose_paddings2=[[1, 0], [0, 1], [0, 1]],
                           ))
          .then(Vectorize(fun_name, ''))
          .then(LoweringOnlyExpert(fun_name,
                                    op_name,
                                    transpose_lowering='eltwise')),
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
      default_problem_sizes_list=[ \
        [parallel_tile_size * num_tiles_per_thread * num_threads, 128, 512]],
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
        torch.set_num_threads(num_threads)
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
