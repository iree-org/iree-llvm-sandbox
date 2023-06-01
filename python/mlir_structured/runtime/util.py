import contextlib
import inspect
import os
import sys
import tempfile
from contextlib import ExitStack
from io import StringIO
from typing import Optional, Callable

from mlir_structured.dialects import scf
from mlir_structured.dialects.indexing import constant, maybe_cast, _update_caller_vars
from mlir_structured.ir import (
    Context,
    Module,
    InsertionPoint,
    Location,
    StringAttr,
    OpView,
)
from mlir_structured.passmanager import PassManager


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
      print("before")
      yield iv, for_op.result
      print("after")
    else:
      yield iv
      scf.YieldOp([])


def scf_yield(*val):
  scf.YieldOp(val)


class IndexingCompilerError(Exception):

  def __init__(self, value: str):
    super().__init__()
    self.value = value

  def __str__(self) -> str:
    return self.value


@contextlib.contextmanager
def disable_multithreading(context=None):
  if context is None:
    context = Context.current

  context.enable_multithreading(False)
  yield
  context.enable_multithreading(True)


def get_module_name_for_debug_dump(module):
  if not "indexing.debug_module_name" in module.operation.attributes:
    return "UnnammedModule"
  return StringAttr(
      module.operation.attributes["nelli.debug_module_name"]).value


def run_pipeline(
    module,
    pipeline: str,
    description: Optional[str] = None,
    enable_ir_printing=False,
    print_pipeline=False,
):
  """Runs `pipeline` on `module`, with a nice repro report if it fails."""
  module_name = get_module_name_for_debug_dump(module)
  try:
    original_stderr = sys.stderr
    sys.stderr = StringIO()
    # Lower module in place to make it ready for compiler backends.
    with ExitStack() as stack:
      stack.enter_context(module.context)
      asm_for_error_report = module.operation.get_asm(
          large_elements_limit=10,
          enable_debug_info=True,
      )
      pm = PassManager.parse(pipeline)
      if print_pipeline:
        print(pm)
      if enable_ir_printing:
        stack.enter_context(disable_multithreading())
        pm.enable_ir_printing()

      pm.run(module.operation)
  except Exception as e:
    print(e, file=sys.stderr)
    filename = os.path.join(tempfile.gettempdir(), module_name + ".mlir")
    with open(filename, "w") as f:
      f.write(asm_for_error_report)
    debug_options = "-mlir-print-ir-after-all -mlir-disable-threading"
    # Put something descriptive here even if description is empty.
    description = description or f"{module_name} compile"

    message = f"""\
            {description} failed with the following diagnostics:
            
            {'*' * 80}
            {sys.stderr.getvalue().strip()}
            {'*' * 80}

            For developers, the error can be reproduced with:
            $ mlir-opt {debug_options} -pass-pipeline='{pipeline}' {filename}
            """
    trimmed_message = "\n".join([m.lstrip() for m in message.split("\n")])
    raise IndexingCompilerError(trimmed_message) from None
  finally:
    sys.stderr = original_stderr

  return module


def find_ops(op, pred: Callable[[OpView], bool]):
  matching = []

  def find(op):
    for r in op.regions:
      for b in r.blocks:
        for o in b.operations:
          if pred(o):
            matching.append(o)
          find(o)

  find(op)

  return matching
