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

# No tiling.
expert_no_tiling = LoweringOnlyExpert('', '')

# 1 level of tiling.
# Note: SingleTilingExpert fails to vectorize if things don't divide and this
# is reported as a full failure.
expert_tile_1 = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24], peel=[]) \
    .then(Vectorize('matmul', ''))                                   \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tile and interchange.
# Note: SingleTilingExpert fails to vectorize if things don't divide and this
# is reported as a full failure.
expert_tile_and_interchange_1 = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24],
         tile_interchange=[2, 0, 1], peel=[])                        \
    .then(Vectorize('matmul', ''))                                   \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling, peel, scalarize the remaining dynamic dims.
# TODO: scalarize_dyn_dims should be exposed as a variable in Tile transformation
# to enable tuning and pass it into the transformation list directly.
expert_tile_1_peel_scalarize = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8], peel=[0])       \
    .then(Tile('matmul', 'linalg.generic', scalarize_dyn_dims=True)) \
    .then(Vectorize('matmul', 'linalg.generic'))                     \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling, with padding.
expert_tile_1_pad = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24])         \
    .then(Pad('matmul', 'linalg.generic',                           \
              padding_values=[0.0, 0.0, 0.0],                       \
              padding_dimensions=[0, 1, 2],                         \
              pack_paddings=[1, 1, 1]))                             \
    .then(Vectorize('matmul', 'linalg.generic'))                    \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling, with padding hoisted and transposed.
expert_tile_1_pad_hoist = \
    Tile('matmul',
         'linalg.generic',
         tile_sizes=[8, 8, 64]) \
    .then(Pad('matmul',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1, 2],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[3, 3, 3],
              transpose_paddings=[[1, 0], [0, 1], [0, 1]])) \
    .then(Vectorize('matmul', 'linalg.generic'))            \
    .then(LoweringOnlyExpert('', ''))
# 2 levels of tiling, with padding hoisted and transposed.
expert_tile_2_pad_hoist = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24]) \
    .then(Tile('matmul',
                'linalg.generic',
                tile_sizes=[4, 4, 12])) \
    .then(Pad('matmul',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1, 2],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[6, 6, 6],
              transpose_paddings=[[1, 0], [0, 1], [0, 1]])) \
    .then(Vectorize('matmul', 'linalg.generic'))            \
    .then(LoweringOnlyExpert('', ''))
# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
expert_tile_3_pad_hoist_peel = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24]) \
    .then(Tile('matmul',
                'linalg.generic',
                tile_sizes=[4, 4, 12])) \
    .then(Pad('matmul',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1, 2],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[6, 6, 6]))                 \
    .then(Tile('matmul',
               'linalg.generic',
               tile_sizes=[2, 3, 7],
               peel=[0, 1, 2]))                            \
    .then(Vectorize('matmul', ''))                         \
    .then(LoweringOnlyExpert('', ''))

# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
# Scalarize remaining dynamic dims.
# FIXME: UnrollOneVectorOp does not implement build_transform_ir
# .then(UnrollOneVectorOp('matmul',
#                         'vector.contract',
#                         source_shape=[2, 3, 7],
#                         target_shape=[1, 3, 7]))         \
# .then(UnrollOneVectorOp('matmul',
#                         'vector.contract',
#                         source_shape=[2, 3, 5],
#                         target_shape=[1, 3, 5]))         \
expert_tile_3_pad_hoist_peel_scalarize = \
    Tile('matmul', 'linalg.generic', tile_sizes=[8, 8, 24]) \
    .then(Tile('matmul',
                'linalg.generic',
                tile_sizes=[4, 4, 12])) \
    .then(Pad('matmul',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1, 2],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[6, 6, 6])) \
    .then(Tile('matmul',
                'linalg.generic',
                tile_sizes=[2, 3, 7],
                peel=[0, 1, 2]))                                     \
    .then(Tile('matmul', 'linalg.generic', scalarize_dyn_dims=True)) \
    .then(Vectorize('matmul', 'linalg.generic'))                     \
    .then(Vectorize('matmul', ''))                                   \
    .then(LoweringOnlyExpert('', ''))

# Fuse, then tile.
expert_fuse_2_tile_1 = \
    Fuse('matmul', 'linalg.generic', tile_sizes=[8, 16, 0])        \
    .then(Fuse('matmul', 'linalg.generic', tile_sizes=[4, 4, 0]))  \
    .then(Tile('matmul', 'linalg.generic', tile_sizes=[0, 0, 24])) \
    .then(Vectorize('matmul', ''))                                 \
    .then(LoweringOnlyExpert('', ''))

# FIXME: could not find replacement for tracked op (failed to apply:
#       %7 = iree_linalg_transform.outline_loop %6 {func_name = "foo"})
# .then(OutlineOneParentLoop('matmul',
#                            'vector.contract',
#                            'foo',
#                            parent_loop_num=2))         \
expert_fuse_and_pad = \
    Fuse('matmul', 'linalg.generic', tile_sizes=[16, 16, 0]) \
    .then(Tile('matmul',
         'linalg.generic',
         tile_sizes=[8, 8, 32])) \
    .then(Pad('matmul',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1, 2],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[3, 3, 3]))                     \
    .then(Vectorize('matmul', 'linalg.generic'))             \
    .then(Tile('matmul', 'linalg.fill', tile_sizes=[8, 8]))  \
    .then(Vectorize('matmul', 'linalg.fill'))                \
    .then(Vectorize('matmul', 'linalg.generic'))             \
    .then(LoweringOnlyExpert('', ''))

expert_fuse_and_pad_and_pipeline = \
    Tile('matmul', 'linalg.generic', tile_sizes=[4, 4, 4]) \
    .then(Vectorize('matmul', ''))           \
    .then(PipelineOneParentLoop('matmul',
                                'vector.contract',
                                parent_loop_num=1,
                                II=10,
                                read_latency=20))          \
    .then(LoweringOnlyExpert('', ''))

all_experts = [
    e.print_ir(after_all=False) for e in [ \
        expert_no_tiling,
        expert_tile_1,
        expert_tile_and_interchange_1,
        expert_tile_1_peel_scalarize,
        expert_tile_1_pad,
        expert_tile_1_pad_hoist,
        expert_tile_2_pad_hoist,
        # Input size is too small for 3 levels of tiling.
        #expert_tile_3_pad_hoist_peel,
        #expert_tile_3_pad_hoist_peel_scalarize,
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
