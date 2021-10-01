# RUN: %PYTHON %s 2>&1 | FileCheck %s

import sys, time
from typing import List
from collections import namedtuple
from collections.abc import Callable
from itertools import chain
from typing import Sequence, Optional

import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects.linalg.opdsl.lang import OperandKind
from mlir.dialects import scf
from mlir.dialects import std
from mlir.execution_engine import *
from mlir.runtime import *

from .transforms import *

f16 = "f16"
f32 = "f32"
f64 = "f64"

numpy_types = {f16: np.float16, f32: np.float32, f64: np.float64}

scalar_types = list(numpy_types.keys())


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


def attach_inplaceable_attributes(func: builtin.FuncOp, rank: int,
                                  inplaceable: Sequence[Optional[bool]]):
  identity_map = AffineMapAttr.get(AffineMap.get_identity(rank))
  read_attrs = DictAttr.get({
      "linalg.inplaceable": BoolAttr.get(False),
      "linalg.buffer_layout": identity_map
  })
  write_attrs = DictAttr.get({
      "linalg.inplaceable": BoolAttr.get(True),
      "linalg.buffer_layout": identity_map
  })
  func.arg_attrs = ArrayAttr.get([
      DictAttr.get({}) if flag is None else write_attrs if flag else read_attrs
      for flag in inplaceable
  ])


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
  func.attributes["passthrough"] = ArrayAttr.get(attributes)


def emit_main_function(operand_defs, func_name: str, **assignments):
  inputs = [
      odef for odef in operand_defs
      if odef.kind == OperandKind.InputTensor or odef.kind == OperandKind.Scalar
  ]
  outputs = [
      odef for odef in operand_defs if odef.kind == OperandKind.OutputTensor
  ]

  assert (len(outputs) == 1)
  return_types = [operand_type(output, **assignments) for output in outputs]
  param_types = [operand_type(operand, **assignments) for operand in inputs]
  param_types += return_types
  func = builtin.FuncOp("main", (param_types + [IndexType.get()], return_types))

  attach_inplaceable_attributes(
      func,
      rank=2,
      inplaceable=[False] * len(inputs) + [True] * len(outputs) + [None])
  func.attributes["llvm.emit_c_interface"] = UnitAttr.get()

  index_type = IndexType.get()
  with InsertionPoint(func.add_entry_block()):
    constant_zero = std.ConstantOp(index_type, IntegerAttr.get(index_type, 0))
    constant_one = std.ConstantOp(index_type, IntegerAttr.get(index_type, 1))
    loop = scf.ForOp(constant_zero.result, func.arguments[-1],
                     constant_one.result, [func.arguments[2]])
    with InsertionPoint(loop.body):
      call = std.CallOp(return_types, FlatSymbolRefAttr.get(func_name),
                        func.arguments[:-2] + loop.inner_iter_args)
      scf.YieldOp(call.results)
    std.ReturnOp(loop.results)


def build_op_under_context_manager(op, transform: Callable, **assignments):
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
      # TODO: in the future, should be writeable more concisely as:
      #   zero = std.constant(0.0, elem_type)
      #   tmp = linalg.fill(zero, out)
      #   linalg.matmul(lhs, rhs, tmp)
      zero = std.ConstantOp(
          value=FloatAttr.get(return_elem_type, 0.),
          result=return_elem_type).result
      tensor_zero = linalg.FillOp(output=outer_args[-1], value=zero).results[0]
      args = outer_args[:-1]
      return op(*args, outs=[tensor_zero])

  # Set the bufferization and optimization attributes.
  func = module.operation.regions[0].blocks[0].operations[0]
  attach_inplaceable_attributes(func, 2, [False, False, True])
  attach_passthrough(func, avx512=True)

  # JIT compile.
  start = time.time()
  with InsertionPoint(module.body):
    emit_main_function(operand_defs, "matmul_on_tensors", **assignments)
  transformed_module = transform("matmul_on_tensors", module)
  execution_engine = ExecutionEngine(transformed_module)
  elapsed_compilation_s = time.time() - start

  return transformed_module, execution_engine


def compile_and_callback(op, transform: Callable, callback: Callable,
                         **assignments):
  with Context() as ctx, Location.unknown():
    module, execution_engine = build_op_under_context_manager(
        op, transform, **assignments)
    callback(module, execution_engine)
