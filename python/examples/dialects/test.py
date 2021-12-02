import os

import mlir.dialects.linalg_ext as linalg_ext

assert linalg_ext.YieldOp is not None, \
    "Could not load linalg_ext.YieldOp"

