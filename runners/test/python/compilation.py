# RUN: %PYTHON %s 2>&1 | FileCheck %s

import sys, time
from typing import List
from collections import namedtuple
from collections.abc import Callable
from itertools import chain

import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.execution_engine import *
from mlir.runtime import *

from transforms import *

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


def scalar_type(tdef, **assignments):
  return mlir_type(assignments[tdef.type_var.name])


def ranked_tensor_type(tdef, **assignments):
  shape = tuple(assignments[sym.symname] for sym in tdef.shape)
  return RankedTensorType.get(shape, scalar_type(tdef, **assignments))


def op_boilerplate(op, func_name: str, **assignments):
  assert (len(op.model.outputs) == 1)
  return_type = str(ranked_tensor_type(op.model.outputs[0], **assignments))

  param_types = []
  for tdef in chain(op.model.inputs, op.model.outputs):
    param_types.append(str(ranked_tensor_type(tdef, **assignments)))

  letter = lambda i: chr(ord("A") + i)
  param_names = [f"%{letter(i)}" for (i, _) in enumerate(param_types)]
  params = ", ".join(
      f"{name} : {ty}" for (name, ty) in zip(param_names, param_types))
  writeable_params = [False] * len(op.model.inputs) + [True] * len(
      op.model.outputs)
  writeable_params = ['"true"' if p else '"false"' for p in writeable_params]
  writeable_params = ", ".join(writeable_params)

  in_args = ", ".join(param_names[:-1])
  out_arg = param_names[-1]
  iter_arg = "%iter" + out_arg[1:]

  return f"""
func @main({params}, %iters : index)
  -> {return_type}
  attributes {{
      llvm.emit_c_interface,
// Activat manually for AVX-512.
   passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]],
// Manually set up `__writeable_func_buffer_args_attr__` to allow proper inplace
// behavior. This is akin to a user annotation that the compiler understands.
      __writeable_func_buffer_args_attr__ = [{writeable_params}] }}
{{
  %c0 = constant 0: index
  %c1 = constant 1: index

  %res = scf.for %arg0 = %c0 to %iters step %c1 iter_args({iter_arg} = {out_arg}) -> ({return_type}) {{
    %r = call @{func_name}({in_args}, {iter_arg}) :
      ({", ".join(param_types)}) -> ({return_type})
    scf.yield %r : {return_type}
  }}

  return %res : {return_type}
}}
"""


def build_op_under_context_manager(op, transform: Callable, **assignments):
  # Build module and function to benchmark.
  ranked_tensor_types = [
      ranked_tensor_type(td, **assignments)
      for td in chain(op.model.inputs, op.model.outputs)
  ]
  return_elem_type = scalar_type(op.model.outputs[0], **assignments)
  module = Module.create()
  with InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(*ranked_tensor_types)
    def matmul_on_tensors(*outer_args):
      # TODO: in the future, should be writeable more concisely as:
      #   zero = std.constant(0.0, elem_type)
      #   tmp = linalg.fill(out, zero)
      #   linalg.matmul(lhs, rhs, tmp)
      zero = std.ConstantOp(
          value=FloatAttr.get(return_elem_type, 0.),
          result=return_elem_type).result
      tensor_zero = linalg.FillOp(output=outer_args[-1], value=zero).results[0]
      args = outer_args[:-1]
      return op(*args, outs=[tensor_zero])

  # JIT compile.
  start = time.time()
  execution_engine = ExecutionEngine(
      transform(module, op_boilerplate(op, "matmul_on_tensors", **assignments)))
  elapsed_compilation_s = time.time() - start
  print(f"compilation in {elapsed_compilation_s:.{4}}s")

  return execution_engine


def compile_and_callback(op, transform: Callable, callback: Callable,
                         **assignments):
  with Context() as ctx, Location.unknown():
    execution_engine = build_op_under_context_manager(op, transform,
                                                      **assignments)
    callback(execution_engine)
