# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

################################################################################
### Compilation strategies.
################################################################################

# No tiling.
expert_no_tiling = LoweringOnlyExpert('', '')

# 1 level of tiling.
expert_tile_1 = \
    Tile('matvec', 'linalg.generic', tile_sizes=[8, 24]) \
    .then(Vectorize('matvec', ''))                       \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tile and interchange.
expert_tile_and_interchange_1 =       \
    Tile('matvec',
         'linalg.generic',
         tile_sizes=[8, 24],
         tile_interchange=[1, 0])     \
    .then(Vectorize('matvec', ''))    \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling and then generalize and interchange.
expert_tile_1_and_generalize_interchange = \
    Tile('matvec',
         'linalg.generic',
         tile_sizes=[8, 24],
         tile_interchange=[1, 0])                             \
    .then(Generalize('matvec', 'linalg.generic'))             \
    .then(Interchange('matvec', iterator_interchange=[0, 1])) \
    .then(Vectorize('matvec', ''))                            \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling, peel, scalarize the remaining dynamic dims.
expert_tile_1_peel_scalarize = \
    Tile('matvec', 'linalg.generic', tile_sizes=[8], peel=[0], scalarize_dyn_dims=True)       \
    .then(Vectorize('matvec', ''))                                   \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling, with padding.
expert_tile_1_pad = \
    Tile('matvec',
         'linalg.generic',
         tile_sizes=[8, 24])            \
    .then(Pad('matvec',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1],
              pack_paddings=[1, 1, 1])) \
    .then(Vectorize('matvec', ''))      \
    .then(LoweringOnlyExpert('', ''))

# 1 level of tiling, with padding, hoisted.
expert_tile_1_pad_hoist = \
    Tile('matvec',
         'linalg.generic',
         tile_sizes=[8, 24])             \
    .then(Pad('matvec',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[3, 3, 3])) \
    .then(Vectorize('matvec', ''))       \
    .then(LoweringOnlyExpert('', ''))

# 2 levels of tiling, with padding, hoisted.
expert_tile_2_pad_hoist = \
    Tile('matvec', 'linalg.generic', tile_sizes=[8, 24]) \
    .then(Tile('matvec',
            'linalg.generic',
            tile_sizes=[4, 12])) \
    .then(Pad('matvec',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[6, 6, 6])) \
    .then(Vectorize('matvec', ''))       \
    .then(LoweringOnlyExpert('', ''))

# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
expert_tile_3_pad_hoist_peel = \
    Tile('matvec', 'linalg.generic', tile_sizes=[8, 24]) \
    .then(Tile('matvec',
         'linalg.generic',
         tile_sizes=[4, 12])) \
    .then(Pad('matvec',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[6, 6, 6])) \
    .then(Tile('matvec', 'linalg.generic', tile_sizes=[2, 7], peel=[0, 1]))\
    .then(Vectorize('matvec', ''))    \
    .then(LoweringOnlyExpert('', ''))

# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
# Scalarize remaining dynamic dims.
expert_tile_3_pad_hoist_peel_scalarize = \
    Tile('matvec', 'linalg.generic', tile_sizes=[8, 24])\
    .then(Tile('matvec',
         'linalg.generic',
         tile_sizes=[4, 12])) \
    .then(Pad('matvec',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[6, 6, 6]))\
    .then(Tile('matvec', 'linalg.generic', tile_sizes=[2, 7], peel=[0, 1], scalarize_dyn_dims=True)) \
    .then(Vectorize('matvec', ''))    \
    .then(LoweringOnlyExpert('', ''))

# Fuse, then tile.
expert_fuse_2_tile_1 = \
    Fuse('matvec', 'linalg.generic', tile_sizes=[8, 16])       \
    .then(Fuse('matvec', 'linalg.generic', tile_sizes=[4, 4])) \
    .then(Tile('matvec', 'linalg.generic', tile_sizes=[2, 3])) \
    .then(Vectorize('matvec', ''))    \
    .then(LoweringOnlyExpert('', ''))

expert_fuse_and_pad = \
    Fuse('matvec', 'linalg.generic', tile_sizes=[16, 16]) \
    .then(Tile('matvec',
               'linalg.generic',
               tile_sizes=[8, 12])) \
    .then(Pad('matvec',
              'linalg.generic',
              padding_values=[0.0, 0.0, 0.0],
              padding_dimensions=[0, 1],
              pack_paddings=[1, 1, 1],
              hoist_paddings=[3, 3, 3])) \
    .then(Vectorize('matvec', 'linalg.generic')) \
    .then(Tile('matvec', 'linalg.fill', tile_sizes=[8, 8]))\
    .then(Vectorize('matvec', ''))    \
    .then(LoweringOnlyExpert('', ''))


all_experts = [ \
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
    expert_fuse_and_pad
              ]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[24, 32], [27, 37]]
  test_harness(lambda s, t: EinsumProblem('mn,n', 'mn', 2), [[np.float32] * 3],
               test_sizes(keys, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name='matvec')


if __name__ == '__main__':
  main()
