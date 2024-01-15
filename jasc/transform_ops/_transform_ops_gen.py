# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Trampoline to run generated MLIR Python code.

Generated tablegen dialects expect to be able to find some symbols from the
mlir.dialects package.
"""

from jaxlib.mlir.dialects._transform_ops_gen import _Dialect
