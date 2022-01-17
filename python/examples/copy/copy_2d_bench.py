# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *
from ..core.utils import *

from ..contraction.definitions import EinsumProblem

from typing import List

base_fun_name = 'copy_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################


def all_experts(fun_name: str, problem_sizes: List[int]):
  sizes1 = l1_2d_divisible_tile_sizes(problem_sizes)
  sizes_for_register_tiling = [ \
    ts if ts > 0 else s for (s, ts) in zip(problem_sizes, sizes1) \
  ]
  sizes2 = register_2d_divisible_tile_sizes(sizes_for_register_tiling)

  # Before bufferization, the IR only has a tensor.extract_slice /
  #   tensor.insert_slice pair.
  # Bufferization then properly introduces linalg.copy ops.
  # We want to make more these `linalg.copy` more efficient.
  # In the case of a single copy benchmark it is the one true thing to optimize.
  return [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=False, at_begin=False, llvm=False) for e in [         \
      Tile(fun_name=fun_name,
            op_name=op_name,
            tile_sizes=sizes2)
      .then(Bufferize())
      .then(Vectorize(fun_name=fun_name, op_name='linalg.copy'))
      .then(LowerVectors())
      .then(LowerToLLVM())
    ]
  ]


################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']

copy_2D_perf_search_list = [
    [32, 64],
    [int(112 / 2) * int(112 / 2), 32 * 4],  # approx. depthwise_conv_2d size
]


# CHECK-NOT: FAILURE
def main():
  n_iters = 10000
  for problem_sizes in copy_2D_perf_search_list:
    fun_name = base_fun_name + '_offset_0' + \
          '_sizes' + ''.join('_' + str(sz) for sz in problem_sizes) + \
          '_strides_' + str(problem_sizes[1]) + '_1'
    test_harness(lambda s, t: EinsumProblem('mn->mn', 0.0),
                 [[np.float32] * 2],
                 test_sizes(keys, [problem_sizes]),
                 all_experts(fun_name, problem_sizes),
                 n_iters=n_iters,
                 function_name=fun_name,
                 dump_ir_to_file='/tmp/abc.mlir',
                 dump_obj_to_file='/tmp/abc.o')


if __name__ == '__main__':
  main()
