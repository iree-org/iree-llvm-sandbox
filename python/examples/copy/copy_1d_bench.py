# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *
from ..core.utils import *

from .definitions import *
from .ops import *

from typing import List

base_fun_name = 'copy_1d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################


# Before bufferization, the IR only has a tensor.extract_slice /
#   tensor.insert_slice pair.
# Bufferization then properly introduces linalg.copy ops.
# We want to make more these `linalg.copy` more efficient.
# In the case of a single copy benchmark it is the one true thing to optimize.
def all_experts(fun_name: str):
  return [
    # Note: `\` char at the end of next line prevents formatter reflows, keep it.
    e.print_ir(after_all=True, at_begin=False, llvm=False) for e in [         \
      Tile(fun_name=fun_name,
            op_name=op_name,
            tile_sizes=[16])
      .then(Bufferize())
      .then(Vectorize(fun_name=fun_name, op_name='linalg.copy'))
      .then(LowerVectors())
      .then(LowerToLLVM())
    ]
  ]


################################################################################
### Problem instantiations.
################################################################################

keys = ['M']

copy_1D_perf_search_list = [
    [200 * 16],  # sweet spot for prefetchers
]


# CHECK-NOT: FAILURE
def main():
  n_iters = 10000
  for problem_sizes in copy_1D_perf_search_list:
    fun_name = base_fun_name + '_offset_0' + \
          '_sizes' + ''.join('_' + str(sz) for sz in problem_sizes)
    test_harness(lambda s, t: CopyNDProblem(rank=1, op_builder=copy_1d),
                 [[np.float32] * 2],
                 test_sizes(keys, [problem_sizes]),
                 all_experts(fun_name),
                 n_iters=n_iters,
                 function_name=fun_name,
                 dump_ir_to_file='/tmp/abc.mlir',
                 dump_obj_to_file='/tmp/abc.o')


if __name__ == '__main__':
  main()
