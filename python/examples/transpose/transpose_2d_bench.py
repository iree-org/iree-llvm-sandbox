# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains small benchmarks with reasonably-sized problem/tiling sizes
# and codegen options.

from typing import Any, List, Optional, Sequence

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import EinsumProblem

fun_name = 'transpose_2d_on_tensors'
op_name = 'linalg.generic'

################################################################################
### Compilation strategies.
################################################################################


def all_experts(problem_sizes: List[int], transpose_avx2_lowering):
  sizes1 = l1_2d_divisible_tile_sizes(problem_sizes)
  sizes_for_register_tiling = [ \
    ts if ts > 0 else s for (s, ts) in zip(problem_sizes, sizes1) \
  ]
  sizes2 = register_2d_divisible_tile_sizes(sizes_for_register_tiling)

  tile1 = TileAndDecompose(fun_name, op_name, tile_sizes=sizes2)
  tile2 = DoubleTileAndDecompose(fun_name=fun_name,
                                 op_name=op_name,
                                 tile_sizes1=sizes1,
                                 tile_sizes2=sizes2,
                                 pack_paddings2=[0, 1],
                                 hoist_paddings2=[2, 2])
  vectorize = Vectorize(fun_name, op_name)
  lowering = LoweringOnlyExpert(fun_name,
                                op_name,
                                transpose_lowering='shuffle',
                                transpose_avx2_lowering=transpose_avx2_lowering)

  # Compute the expert names.
  tile1_str = str.format(f"{sizes1[0]}x{sizes1[1]}")
  tile2_str = str.format(f"{sizes2[0]}x{sizes2[1]}")
  avx_str = "AVX2" if transpose_avx2_lowering else ""
  all_names = [
    str.format(f"Transpose2D{tile2_str}{avx_str}Expert"),
    str.format(f"Transpose2D{tile1_str}{tile2_str}{avx_str}Expert")
  ]

  # Compute the experts.
  all_experts = [e.print_ir(after_all=False)
      for e in [\
          tile1 + vectorize + lowering,\
          tile2 + vectorize + lowering,
               ]]

  return dict(zip(all_names, all_experts))

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']


# CHECK-NOT: FAILURE
def main():
  # Specify default configuration and parse command line.
  args = test_argparser(
    "transpose 2d benchmark",
    default_n_iters=1000,
    default_problem_sizes_list=[
      # The objective of these problem sizes is to run an experiment where we
      # control register tile sizes to stress test different implementations of
      # vector.transpose while isolating the cases with boundary conditions.
      # The setup is such that our volume of data is close from 256^2, 512^2 and
      # 1024^2.

      ### Second dimension is a multiple of 8 but not 16.
      # First dimension is a multiple of 4 but not 6, 8, 12, 16.
      [4 * 65, 8 * 35],  # ca. 256^2
      # First dimension is a multiple of 6 but not 8, 12, 16
      [6 * 45, 8 * 35],  # ca. 256^2
      # First dimension is a multiple of 8 but not 12 or 16
      [8 * 35, 8 * 35],  # ca. 256^2
      # First dimension is a multiple of 12 but not 16
      [12 * 25, 8 * 35],  # ca. 256^2
      # First dimension is a multiple of 16
      [16 * 16, 8 * 35],  # ca. 256^2
      ### Second dimension is a multiple of 16.
      [4 * 65, 256],  # ca. 256^2
      [6 * 45, 256],  # ca. 256^2
      [8 * 35, 256],  # ca. 256^2
      [12 * 25, 256],  # ca. 256^2
      [16 * 16, 256],  # ca. 256^2

      ### Second dimension is a multiple of 8 but not 16.
      [4 * 145, 8 * 65],  # ca. 512^2
      [6 * 95, 8 * 65],  # ca. 512^2
      [8 * 65, 8 * 65],  # ca. 512^2
      [12 * 55, 8 * 65],  # ca. 512^2
      [16 * 32, 8 * 65],  # ca. 512^2
      ### Second dimension is a multiple of 16.
      [4 * 145, 512],  # ca. 512^2
      [6 * 95, 512],  # ca. 512^2
      [8 * 65, 512],  # ca. 512^2
      [12 * 55, 512],  # ca. 512^2
      [16 * 32, 512],  # ca. 512^2

      # Same idea for 1024^2.
      ### Second dimension is a multiple of 8 but not 16.
      [4 * 275, 8 * 145],  # ca. 1024^2
      [6 * 175, 8 * 145],  # ca. 1024^2
      [8 * 145, 8 * 145],  # ca. 1024^2
      [12 * 85, 8 * 145],  # ca. 1024^2
      [16 * 65, 8 * 145],  # ca. 1024^2
      ### Second dimension is a multiple of 16.
      [4 * 275, 1024],  # ca. 1024^2
      [6 * 175, 1024],  # ca. 1024^2
      [8 * 145, 1024],  # ca. 1024^2
      [12 * 85, 1024],  # ca. 1024^2
      [16 * 65, 1024],  # ca. 1024^2

      # TODO: this is too slow atm.
      # [4096, 4096],
      # [6912, 4608],
    ],
    default_expert_list=[],
    default_dynamic_at_compile_time_list=[],
    default_spec_list=[])

  for problem_sizes in args.problem_sizes_list:
    experts = all_experts(problem_sizes, transpose_avx2_lowering=False)
    experts.update(all_experts(problem_sizes, transpose_avx2_lowering=True))

    test_harness(lambda s, t: EinsumProblem('nm->mn', 0.0),
                 [[np.float32] * 2],
                 test_sizes(keys, [problem_sizes]),
                 experts,
                 n_iters=args.n_iters,
                 function_name=fun_name,
                 dump_ir_to_file='/tmp/abc.mlir',
                 dump_obj_to_file='/tmp/abc.o',
                 dump_data_to_file=args.dump_data)


if __name__ == '__main__':
  main()
