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
    try:
      context = Context.current
    except ValueError as e:
      assert str(e) == "No current Context"
      context = Context()
  if location is None:
    location = Location.unknown(context=context)
  with context, location:
    if src is not None:
      module = Module.parse(src, context=context)
    else:
      module = Module.create(loc=location)
    with InsertionPoint(module.body):
      yield module
