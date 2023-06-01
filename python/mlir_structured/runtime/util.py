import contextlib
import inspect
from typing import Optional

from mlir_structured.ir import (
    Context,
    Module,
    InsertionPoint,
    Location,
)
from mlir_structured.dialects.indexing import constant, maybe_cast, _update_caller_vars
from mlir_structured.dialects import scf


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


def scf_range(start, stop=None, step=1, iter_args=None):
  if iter_args is None:
    iter_args = []
  if stop is None:
    stop = start
    start = 0

  if isinstance(start, int):
    start = constant(start, index=True)
  if isinstance(stop, int):
    stop = constant(stop, index=True)
  if isinstance(step, int):
    step = constant(step, index=True)
  for_op = scf.ForOp(start, stop, step, iter_args)
  iv = maybe_cast(for_op.induction_variable)
  with InsertionPoint(for_op.body):
    if len(iter_args):
      previous_frame = inspect.currentframe().f_back
      _update_caller_vars(previous_frame, iter_args, for_op.inner_iter_args)
      yield iv, for_op.result
    else:
      yield iv
      scf.YieldOp([])


def scf_yield(*val):
  scf.YieldOp(val)
