"""Trampoline to run generated MLIR Python code.

Generated tablegen dialects expect to be able to find some symbols from the
mlir.dialects package.
"""

from jaxlib.mlir.dialects._transform_ops_gen import _Dialect
