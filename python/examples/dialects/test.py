import mlir.dialects.iree_linalg_ext as iree_linalg_ext

assert hasattr(iree_linalg_ext,
               'YieldOp'), 'Could not load iree_linalg_ext.YieldOp'
