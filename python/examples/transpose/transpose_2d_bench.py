# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

fun_name = 'transpose_2d'
op_name = 'linalg.generic'

################################################################################
# Compilation strategies.
################################################################################

# Note: `\` char at the end of next line prevents formatter reflows, keep it.
all_names = [  \
  "SingleTiling2DPeel" \
  ]

all_experts = [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [ \
        SingleTilingExpert(
            fun_name,
            op_name,
            #           M  N
            tile_sizes=[8, 8],
            peel=[0, 1])
          .then(Vectorize(fun_name, op_name))
          .then(Bufferize())
          .then(LowerVectors())
          .then(LowerToLLVM()),
    ]
]

################################################################################
# Problem instantiation
################################################################################

keys = ['m', 'n']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
    "transpose 2d benchmark",
    default_n_iters=100,
    #  M   N
    default_problem_sizes_list=[ \
      [8, 8],
      [16, 16],
      [32, 32],
    ],
    default_expert_list=all_names,
    default_dynamic_at_compile_time_list=[],
    default_spec_list=[])

  test_harness(lambda sizes, t: EinsumProblem('nm->mn', 'mn', 0), \
               [[np.float32] * 2],
               test_sizes(keys, args.problem_sizes_list),
               test_experts(all_experts, all_names, args.expert_list),
               n_iters=args.n_iters,
               function_name=fun_name,
               dump_ir_to_file='/tmp/abcd.mlir',
               dump_obj_to_file='/tmp/abcd.o',
               dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
