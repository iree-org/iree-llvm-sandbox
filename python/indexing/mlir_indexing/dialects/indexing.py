#  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
#  See https://llvm.org/LICENSE.txt for license information.
#  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import atexit
import contextlib
import ctypes
import sys
from typing import Optional, Sequence


# this is specifically for _ZN4mlir6python23DefaultingPyMlirContext7resolveEv
@contextlib.contextmanager
def dl_open_guard():
  old_flags = sys.getdlopenflags()
  sys.setdlopenflags(old_flags | ctypes.RTLD_GLOBAL)
  yield
  sys.setdlopenflags(old_flags)


with dl_open_guard():
  # noinspection PyUnresolvedReferences
  from .._mlir_libs import _mlir
  from ..ir import (register_attribute_builder, DenseI64ArrayAttr, Context,
                    Location, Module, InsertionPoint)

from ._indexing_ops_gen import *
from .._mlir_libs._indexingDialects import *

DefaultContext = Context()
# Push a default context onto the context stack at import time.
DefaultContext.__enter__()
DefaultContext.allow_unregistered_dialects = False


@atexit.register
def __exit_ctxt():
  DefaultContext.__exit__(None, None, None)


register_dialect(DefaultContext)


@contextlib.contextmanager
def mlir_mod_ctx(src: Optional[str] = None):
  with Location.unknown() as loc:
    if src is not None:
      module = Module.parse(src)
    else:
      module = Module.create(loc=loc)
    with InsertionPoint(module.body):
      yield module


@register_attribute_builder("DenseI64ArrayAttr")
def get_dense_int64_array_attr(
    values: Sequence[int],
    context: Optional[Context] = None) -> DenseI64ArrayAttr:
  if context is None:
    context = DefaultContext
  if values is None:
    return DenseI64ArrayAttr.get([], context)
  return DenseI64ArrayAttr.get(values, context)
