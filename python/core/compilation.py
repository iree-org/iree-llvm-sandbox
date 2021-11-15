# RUN: %PYTHON %s 2>&1 | FileCheck %s
# pytype: skip-file

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

f16 = "f16"
f32 = "f32"
f64 = "f64"

numpy_types = {f16: np.float16, f32: np.float32, f64: np.float64}

scalar_types = list(numpy_types.keys())

_MLIR_RUNNER_UTILS_LIB_ENV = "MLIR_RUNNER_UTILS_LIB"
_MLIR_RUNNER_UTILS_LIB_DEFAULT = "libmlir_runner_utils.so"

def numpy_type(scalar_type):
  numpy_types[scalar_type]


def mlir_type(scalar_type):
  if scalar_type == f16:
    return F16Type.get()
  elif scalar_type == f32:
    return F32Type.get()
  elif scalar_type == f64:
    return F64Type.get()
  else:
    raise Exception(f"unknown scalar type: {scalar_type}")


def scalar_type(odef, **assignments):
  return mlir_type(assignments[odef.type_var.name])


def operand_type(odef, **assignments):
  if odef.kind == OperandKind.Scalar:
    return scalar_type(odef, **assignments)
  if (odef.kind == OperandKind.InputTensor or
      odef.kind == OperandKind.OutputTensor):
    shape = tuple(assignments[sym.symname] for sym in odef.size_exprs)
    return RankedTensorType.get(shape, scalar_type(odef, **assignments))
  raise Exception(f"unsupported operand type: {repr(odef)}")


def attach_inplaceable_attributes(func: builtin.FuncOp,
                                  inplaceable: Sequence[Optional[bool]]):
  attrs = []
  for t, flag in zip(func.type.inputs, inplaceable):
    if flag is None:
      attrs.append(DictAttr.get({}))
      continue
    assert RankedTensorType.isinstance(t), "Not a RankedTensorType: {t}"
    identity_map = AffineMapAttr.get(
        AffineMap.get_identity(RankedTensorType(t).rank))
    attrs.append(
        DictAttr.get({
            "linalg.inplaceable": BoolAttr.get(flag),
            "linalg.buffer_layout": identity_map
        }))
  func.arg_attrs = attrs


def attach_passthrough(func: builtin.FuncOp,
                       extras: Sequence[Attribute] = [],
                       avx512: bool = False):
  attributes = extras[:]
  if avx512:
    attributes.append(
        ArrayAttr.get(
            [StringAttr.get("target-cpu"),
             StringAttr.get("skylake-avx512")]))
    attributes.append(
        ArrayAttr.get(
            [StringAttr.get("prefer-vector-width"),
             StringAttr.get("512")]))
  else:
    attributes.append(
        ArrayAttr.get(
            [StringAttr.get("target-cpu"),
             StringAttr.get("broadwell")]))
    attributes.append(
        ArrayAttr.get(
            [StringAttr.get("prefer-vector-width"),
             StringAttr.get("256")]))
  func.attributes["passthrough"] = ArrayAttr.get(attributes)


def emit_benchmarking_function(name: str,
                               func: builtin.FuncOp) -> builtin.FuncOp:
  """Produces the benchmarking function.

  This function calls the given function `func` as many times as requested by
  its last argument.
  """
  i64_type = IntegerType.get_signless(64)
  nano_time = builtin.FuncOp(
      "nano_time", ([], [i64_type]), visibility="private")
  nano_time.attributes["llvm.emit_c_interface"] = UnitAttr.get()

  wrapper = builtin.FuncOp(
      name, (func.arguments.types + [IndexType.get()],
             func.type.results + [i64_type]),
      visibility="public")
  wrapper.attributes["llvm.emit_c_interface"] = UnitAttr.get()
  wrapper.arg_attrs = func.arg_attrs + [DictAttr.get()]

  num_results = len(func.type.results)
  with InsertionPoint(wrapper.add_entry_block()):
    zero = arith.ConstantOp.create_index(0)
    one = arith.ConstantOp.create_index(1)
    total_time = arith.ConstantOp(i64_type, 0)
    iter_args = list(wrapper.arguments[-num_results - 1:-1])
    iter_args.append(total_time.result)
    loop = scf.ForOp(zero, wrapper.arguments[-1], one, iter_args)
    with InsertionPoint(loop.body):
      time_accumulator = loop.inner_iter_args[-1]
      start = std.CallOp(nano_time, [])
      call = std.CallOp(
          func,
          wrapper.arguments[:-num_results - 1] + loop.inner_iter_args[:-1])
      end = std.CallOp(nano_time, [])
      time = arith.SubIOp(end, start)
      partial_time = arith.AddIOp(time_accumulator, time)
      scf.YieldOp(list(call.results) + [partial_time.result])
    std.ReturnOp(loop)

  return wrapper


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


# JIT compile and return an execution engine that can be invoked.
# Needs to be run under Context.
def compile_to_execution_engine(module,
                                transform: Callable,
                                opt_level: int = 3):
  start = time.time()
  transformed_module = transform(module)
  execution_engine = ExecutionEngine(
      transformed_module,
      opt_level,
      shared_libs=[
          os.getenv(_MLIR_RUNNER_UTILS_LIB_ENV, _MLIR_RUNNER_UTILS_LIB_DEFAULT)
      ])
  elapsed_compilation_s = time.time() - start
  print(f"compilation in {elapsed_compilation_s:.{4}}s")
  return transformed_module, execution_engine
