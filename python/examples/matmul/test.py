# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

import typing as tp

################################################################################
### Compilation strategies.
################################################################################


def TestExpert(transforms: tp.Sequence[tp.Union[Transform,
                                                TransformationList]]):
  return (TransformationList(transforms=transforms) + Bufferize() +
          LoweringOnlyExpert('matmul', 'linalg.generic'))


# TODO: Check generate code for basic code quality, e.g., no linalg.copy.

# No tiling.
expert_no_tiling = LoweringOnlyExpert('matmul', 'linalg.generic')

# 1 level of tiling.
expert_tile_1 = SingleTilingExpert('matmul',
                                   'linalg.generic',
                                   tile_sizes=[8, 8, 24],
                                   pad=False,
                                   peel=[])

# 1 level of tile and interchange.
expert_tile_and_interchange_1 = SingleTilingExpert('matmul',
                                                   'linalg.generic',
                                                   tile_sizes=[8, 8, 24],
                                                   tile_interchange=[2, 0, 1],
                                                   pad=False,
                                                   peel=[])

# 1 level of tiling and then generalize and interchange.
expert_tile_1_and_generalize_interchange = \
    Tile.then(Generalize).then(Vectorize).then(LoweringOnlyExpert)(\
      'matmul',                                         \
      'linalg.generic',                                            \
      tile_sizes=[8, 8, 24],                                       \
      tile_interchange=[2, 0, 1],                                  \
      iterator_interchange=[0, 2, 1])

# 1 level of tiling, peel, scalarize the remaining dynamic dims.
# TODO: scalarize_dyn_dims should be exposed as a variable in Tile transformation
# to enable tuning and pass it into the transformation list directly.
expert_tile_1_peel_scalarize = TransformationList(transforms=[
    Tile('matmul', 'linalg.generic', tile_sizes=[8], peel=[0]),
    Tile('matmul', 'linalg.generic', scalarize_dyn_dims=True),
] + Vectorize.then(LoweringOnlyExpert)('matmul', 'linalg.generic').transforms)

# 1 level of tiling, with padding.
expert_tile_1_pad = Tile('matmul',
                         'linalg.generic',
                         tile_sizes=[8, 8, 24],
                         pad=True,
                         pack_paddings=[1, 1, 1]).then(
                             Vectorize('matmul', 'linalg.generic') +
                             LoweringOnlyExpert('matmul', 'linalg.generic'))

# 1 level of tiling, with padding hoisted and transposed.
expert_tile_1_pad_hoist = TestExpert([
    Tile('matmul',
         'linalg.generic',
         tile_sizes=[8, 8, 64],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[3, 3, 3],
         transpose_paddings=[[1, 0], [0, 1], [0, 1]]),
    Vectorize('matmul', 'linalg.generic')
])
# 2 levels of tiling, with padding hoisted and transposed.
expert_tile_2_pad_hoist = TestExpert([
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24]),
    Tile('matmul',
         'linalg.generic',
         tile_sizes=[4, 4, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[6, 6, 6],
         transpose_paddings=[[1, 0], [0, 1], [0, 1]]),
    Vectorize('matmul', 'linalg.generic')
])
# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
expert_tile_3_pad_hoist_peel = TestExpert([
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24]),
    Tile('matmul',
         'linalg.generic',
         tile_sizes=[4, 4, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[6, 6, 6]),
    Tile('matmul', 'linalg.generic', tile_sizes=[2, 3, 7], peel=[0, 1, 2]),
    Vectorize('matmul', 'linalg.generic')
])
# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
# Scalarize remaining dynamic dims.
expert_tile_3_pad_hoist_peel_scalarize = TestExpert([
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24]),
    Tile('matmul',
         'linalg.generic',
         tile_sizes=[4, 4, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[6, 6, 6]),
    Tile('matmul', 'linalg.generic', tile_sizes=[2, 3, 7], peel=[0, 1, 2]),
    Tile('matmul', 'linalg.generic', scalarize_dyn_dims=True),
    Vectorize('matmul', 'linalg.generic'),
    UnrollOneVectorOp(
        'matmul',
        'vector.contract',
        source_shape=[2, 3, 7],
        target_shape=[1, 3, 7],
    ),
    UnrollOneVectorOp(
        'matmul',
        'vector.contract',
        source_shape=[2, 3, 5],
        target_shape=[1, 3, 5],
    ),
])
# Fuse, then tile.
expert_fuse_2_tile_1 = TestExpert([
    Fuse('matmul', 'linalg.generic', tile_sizes=[8, 16, 0]),
    Fuse('matmul', 'linalg.generic', tile_sizes=[4, 4, 0]),
    Tile('matmul', 'linalg.generic', tile_sizes=[0, 0, 24]),
    Vectorize('matmul', 'linalg.generic'),
    Vectorize('matmul', 'linalg.fill')
])
expert_fuse_and_pad = TestExpert([
    Fuse('matmul', 'linalg.generic', tile_sizes=[16, 16, 0]),
    Tile('matmul',
         'linalg.generic',
         tile_sizes=[8, 8, 32],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[3, 3, 3]),
    Vectorize('matmul', 'linalg.generic'),
    OutlineOneParentLoop('matmul', 'vector.contract', 'foo', parent_loop_num=2),
    Tile('matmul', 'linalg.fill', tile_sizes=[8, 8]),
    Vectorize('matmul', 'linalg.fill')
])
expert_fuse_and_pad_and_pipeline = TestExpert([
    Tile('matmul', 'linalg.generic', tile_sizes=[4, 4, 4]),
    Vectorize('matmul', 'linalg.generic'),
    PipelineOneParentLoop('matmul',
                          'vector.contract',
                          parent_loop_num=1,
                          II=10,
                          read_latency=20)
])

all_experts = [
    e.print_ir(after_all=False) for e in [ \
        expert_no_tiling,
        expert_tile_1,
        expert_tile_and_interchange_1,
        expert_tile_1_and_generalize_interchange,
        expert_tile_1_peel_scalarize,
        expert_tile_1_pad,
        expert_tile_1_pad_hoist,
        expert_tile_2_pad_hoist,
        expert_tile_3_pad_hoist_peel,
        expert_tile_3_pad_hoist_peel_scalarize,
        expert_fuse_2_tile_1,
        expert_fuse_and_pad,
        expert_fuse_and_pad_and_pipeline
    ]
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n', 'k']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[24, 32, 48], [27, 37, 43]]

  test_harness(lambda s, t: EinsumProblem('mk,kn', 'mnk', 2),
               [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name='matmul')


if __name__ == '__main__':
  main()
