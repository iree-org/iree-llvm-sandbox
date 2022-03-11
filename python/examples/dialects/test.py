import mlir.dialects.linalg_ext as linalg_ext

assert hasattr(linalg_ext, 'YieldOp'), 'Could not load linalg_ext.YieldOp'

from mlir.dialects import iree_linalg_ext
assert hasattr(iree_linalg_ext,
               'YieldOp'), 'Could not load iree_linalg_ext.YieldOp'
