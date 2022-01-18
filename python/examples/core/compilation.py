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
from mlir.dialects import arith, builtin, linalg, memref, scf, std
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
_MLIR_C_RUNNER_UTILS_LIB_ENV = "MLIR_C_RUNNER_UTILS_LIB"
_MLIR_C_RUNNER_UTILS_LIB_DEFAULT = "libmlir_c_runner_utils.so"
_MLIR_RUNNER_EXTRA_LIBS_ENV = "MLIR_RUNNER_EXTRA_LIBS"


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
  nano_time = builtin.FuncOp("nano_time", ([], [i64_type]),
                             visibility="private")
  nano_time.attributes["llvm.emit_c_interface"] = UnitAttr.get()

  memref_of_i64_type = MemRefType.get([-1], i64_type)
  wrapper = builtin.FuncOp(
      # Same signature and an extra buffer of indices to save timings.
      name,
      (func.arguments.types + [memref_of_i64_type], func.type.results),
      visibility="public")
  wrapper.attributes["llvm.emit_c_interface"] = UnitAttr.get()
  wrapper.arg_attrs = func.arg_attrs + [DictAttr.get()]

  num_results = len(func.type.results)
  with InsertionPoint(wrapper.add_entry_block()):
    timer_buffer = wrapper.arguments[-1]
    zero = arith.ConstantOp.create_index(0)
    n_iterations = memref.DimOp(IndexType.get(), timer_buffer, zero)
    one = arith.ConstantOp.create_index(1)
    iter_args = list(wrapper.arguments[-num_results - 1:-1])
    loop = scf.ForOp(zero, n_iterations, one, iter_args)
    with InsertionPoint(loop.body):
      start = std.CallOp(nano_time, [])
      call = std.CallOp(
          func, wrapper.arguments[:-num_results - 1] + loop.inner_iter_args)
      end = std.CallOp(nano_time, [])
      time = arith.SubIOp(end, start)
      memref.StoreOp(time, timer_buffer, [loop.induction_variable])
      scf.YieldOp(list(call.results))
    std.ReturnOp(loop)

  return wrapper


# JIT compile and return an execution engine that can be invoked.
# Needs to be run under Context.
def compile_to_execution_engine(module,
                                transform: Callable,
                                opt_level: int = 3):
  transformed_module = transform(module)
  shared_libs = [
      os.getenv(_MLIR_RUNNER_UTILS_LIB_ENV, _MLIR_RUNNER_UTILS_LIB_DEFAULT),
      os.getenv(_MLIR_C_RUNNER_UTILS_LIB_ENV, _MLIR_C_RUNNER_UTILS_LIB_DEFAULT)
  ]
  extra_libs = os.getenv(_MLIR_RUNNER_EXTRA_LIBS_ENV)
  if extra_libs is not None:
    shared_libs.append(*(str(extra_libs).split(',')))
  execution_engine = ExecutionEngine(transformed_module,
                                     opt_level,
                                     shared_libs=shared_libs)
  return transformed_module, execution_engine
