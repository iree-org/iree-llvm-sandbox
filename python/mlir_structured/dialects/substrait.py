# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence

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

  def __init__(self, *args, version: Optional[Sequence[int]] = None, **kwargs):
    if version is not None:
      major, minor, patch = version
      for part in ["major", "minor", "patch"]:
        if (part + "_number") in kwargs:
          raise ValueError(
              "'version' and '(major|minor|patch)_number' are mutually exclusive"
          )
      args = (major, minor, patch) + args
    super().__init__(*args, **kwargs)
    self.regions[0].blocks.append()

  @property
  def body(self) -> ir.Block:
    return self.regions[0].blocks[0]

  def to_json(self, pretty: bool = False) -> str:
    return to_json(self.operation, pretty)

  def to_binpb(self) -> str:
    return to_binpb(self.operation)

  def to_textpb(self) -> str:
    return to_textpb(self.operation)


@_ods_cext.register_operation(_Dialect, replace=True)
class PlanRelOp(PlanRelOp):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.regions[0].blocks.append()

  @property
  def body(self) -> ir.Block:
    return self.regions[0].blocks[0]
