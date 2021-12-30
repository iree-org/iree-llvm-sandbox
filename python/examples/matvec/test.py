# RUN: %PYTHON %s 2>&1 | FileCheck %s

# This file contains simple test cases that combine various codegen options.

from ..core.experts import *
from ..core.harness import *
from ..core.transforms import *

from ..contraction.definitions import *

################################################################################
### Compilation strategies.
################################################################################


class TestExpert(TransformationList):

  def __init__(self, tiling_transforms):
    t = tiling_transforms + [Bufferize()] + LoweringOnlyExpert(
        'matvec_on_tensors', 'linalg.generic').transforms
    TransformationList.__init__(self, **{'transforms': t})


# TODO: Check generate code for basic code quality, e.g., no linalg.copy.

# No tiling.
expert_no_tiling = TestExpert([])

# 1 level of tiling.
expert_tile_1 = TestExpert([
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[8, 24]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 1 level of tile and interchange.
expert_tile_and_interchange_1 = TestExpert([
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[8, 24],
         tile_interchange=[1, 0]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 1 level of tiling and then generalize and interchange.
expert_tile_1_and_generalize_interchange = TestExpert([
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[8, 24],
         tile_interchange=[1, 0]),
    Generalize('matvec_on_tensors',
               'linalg.generic',
               iterator_interchange=[0, 1]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 1 level of tiling, peel, scalarize the remaining dynamic dims.
expert_tile_1_peel_scalarize = TestExpert([
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[8], peel=[0]),
    Tile('matvec_on_tensors', 'linalg.generic', scalarize_dyn_dims=True),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 1 level of tiling, with padding.
expert_tile_1_pad = TestExpert([
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[8, 24],
         pad=True,
         pack_paddings=[1, 1, 1]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 1 level of tiling, with padding, hoisted.
expert_tile_1_pad_hoist = TestExpert([
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[8, 24],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[3, 3, 3]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 2 levels of tiling, with padding, hoisted.
expert_tile_2_pad_hoist = TestExpert([
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[8, 24]),
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[4, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[6, 6, 6]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
expert_tile_3_pad_hoist_peel = TestExpert([
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[8, 24], pad=False),
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[4, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[6, 6, 6]),
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[2, 7], peel=[0, 1]),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# 3 levels of tiling, with padding, hoisted. Peeling on the 3rd level.
# Scalarize remaining dynamic dims.
expert_tile_3_pad_hoist_peel_scalarize = TestExpert([
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[8, 24]),
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[4, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[6, 6, 6]),
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[2, 7], peel=[0, 1]),
    Tile('matvec_on_tensors', 'linalg.generic', scalarize_dyn_dims=True),
    Vectorize('matvec_on_tensors', 'linalg.generic')
])
# Fuse, then tile.
expert_fuse_2_tile_1 = TestExpert([
    Fuse('matvec_on_tensors', 'linalg.generic', tile_sizes=[8, 16]),
    Fuse('matvec_on_tensors', 'linalg.generic', tile_sizes=[4, 4]),
    Tile('matvec_on_tensors', 'linalg.generic', tile_sizes=[2, 3]),
    Vectorize('matvec_on_tensors', 'linalg.generic'),
    Vectorize('matvec_on_tensors', 'linalg.fill')
])
expert_fuse_and_pad = TestExpert([
    Fuse('matvec_on_tensors', 'linalg.generic', tile_sizes=[16, 16]),
    Tile('matvec_on_tensors',
         'linalg.generic',
         tile_sizes=[8, 12],
         pad=True,
         pack_paddings=[1, 1, 1],
         hoist_paddings=[3, 3, 3]),
    Vectorize('matvec_on_tensors', 'linalg.generic'),
    Tile('matvec_on_tensors', 'linalg.fill', tile_sizes=[8, 8]),
    Vectorize('matvec_on_tensors', 'linalg.fill')
])

all_experts = [
    expert_no_tiling, expert_tile_1, expert_tile_and_interchange_1,
    expert_tile_1_and_generalize_interchange, expert_tile_1_peel_scalarize,
    expert_tile_1_pad, expert_tile_1_pad_hoist, expert_tile_2_pad_hoist,
    expert_tile_3_pad_hoist_peel, expert_tile_3_pad_hoist_peel_scalarize,
    expert_fuse_2_tile_1, expert_fuse_and_pad
]

################################################################################
### Problem instantiations.
################################################################################

keys = ['m', 'n']


def make_size_list(sizes: Sequence):
  return {k: v for k, v in zip(keys, sizes)}


# CHECK-NOT: FAILURE
def main():
  n_iters = 1
  problem_size_list = [[24, 32], [27, 37]]
  test_harness(lambda s, t: EinsumProblem('mn,n'), [[np.float32] * 3],
               map(make_size_list, problem_size_list),
               all_experts,
               n_iters=n_iters,
               function_name='matvec_on_tensors')


if __name__ == '__main__':
  main()
