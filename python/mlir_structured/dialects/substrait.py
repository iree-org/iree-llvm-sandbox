# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ._substrait_ops_gen import *
from ._substrait_ops_gen import _Dialect
from .._mlir_libs._structuredDialects.substrait import *

try:
  from .. import ir
  from ._ods_common import (
      _cext as _ods_cext,)
except ImportError as e:
  raise RuntimeError("Error loading imports from extension module") from e


@_ods_cext.register_operation(_Dialect, replace=True)
class PlanOp(PlanOp):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.regions[0].blocks.append()

  @property
  def body(self) -> ir.Block:
    return self.regions[0].blocks[0]


@_ods_cext.register_operation(_Dialect, replace=True)
class PlanRelOp(PlanRelOp):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.regions[0].blocks.append()

  @property
  def body(self) -> ir.Block:
    return self.regions[0].blocks[0]
