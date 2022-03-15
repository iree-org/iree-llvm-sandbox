import mlir.dialects.iree_linalg_ext as linalg_ext

assert hasattr(linalg_ext, 'YieldOp'), 'Could not load linalg_ext.YieldOp'

import mlir.dialects.iree_linalg_transform as linalg_transform

assert hasattr(linalg_transform, 'TileToLinalgExtTileOp'), \
  'Could not load linalg_transform.TileToLinalgExtTileOp'
