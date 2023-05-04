import contextlib
from typing import Optional

from mlir_structured.ir import (
    Context,
    Module,
    InsertionPoint,
    Location,
)


@contextlib.contextmanager
def mlir_mod_ctx(src: Optional[str] = None,
                 context: Optional[Context] = None,
                 location: Optional[Location] = None):
  if context is None:
    context = Context()
  if location is None:
    location = Location.unknown()
  with context, location:
    if src is not None:
      module = Module.parse(src)
    else:
      module = Module.create()
    with InsertionPoint(module.body):
      yield module


