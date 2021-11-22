import sys, time
import os
from typing import List
from collections import namedtuple
from collections.abc import Callable
from itertools import chain
from typing import Sequence, Optional

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std
from mlir.dialects.linalg.opdsl.lang import OperandKind
from mlir.execution_engine import *
from mlir.runtime import *

from .transforms import *


# TODO: retire this because it has internal  assumptions about number of
# arguments and what is input/output
def build_op_under_context_manager(op,
                                   transform: Callable,
                                   opt_level: int = 3,
                                   **assignments):
  # Build module and function to benchmark.
  operand_defs = sorted(
      op.model.registered_operands.values(),
      key=lambda odef: odef.registered_index)

  ranked_tensor_types = [
      operand_type(odef, **assignments) for odef in operand_defs
  ]
  return_elem_type = scalar_type(operand_defs[-1], **assignments)
  module = Module.create()
  with InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(*ranked_tensor_types)
    def matmul_on_tensors(*outer_args):
      zero = arith.ConstantOp(return_elem_type, 0.0)
      tensor_zero = linalg.FillOp(output=outer_args[-1], value=zero)
      args = outer_args[:-1]
      return op(*args, outs=tensor_zero)

  # Set the bufferization and optimization attributes.
  func = module.operation.regions[0].blocks[0].operations[0]
  attach_inplaceable_attributes(func, [False, False, True])
  attach_passthrough(func, avx512=True)

  # JIT compile.
  start = time.time()
  with InsertionPoint(module.body):
    emit_benchmarking_function("main", func)
  transformed_module = transform("matmul_on_tensors", module)
  execution_engine = ExecutionEngine(
      transformed_module,
      opt_level,
      shared_libs=[
          os.getenv(_MLIR_RUNNER_UTILS_LIB_ENV, _MLIR_RUNNER_UTILS_LIB_DEFAULT)
      ])
  elapsed_compilation_s = time.time() - start

  return transformed_module, execution_engine


# TODO: retire this because build_op_under_context_manager has internal
# assumptions about number of arguments and what is input/output
def compile_and_callback(op, transform: Callable, callback: Callable,
                         **assignments):
  with Context() as ctx, Location.unknown():
    module, execution_engine = build_op_under_context_manager(
        op, transform, **assignments)
    return callback(module, execution_engine)
