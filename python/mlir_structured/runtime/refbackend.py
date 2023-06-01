# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

from __future__ import annotations

import ctypes
import os
from typing import Optional, Callable

import numpy as np

from mlir_structured.execution_engine import ExecutionEngine

from mlir_structured.runtime import (
    UnrankedMemRefDescriptor,
    get_ranked_memref_descriptor,
    get_unranked_memref_descriptor,
)
from mlir_structured.ir import Module, UnitAttr
from mlir_structured.runtime.util import run_pipeline, find_ops


def assert_arg_type_is_supported(ty):
  SUPPORTED = [
      np.float16,
      np.float32,
      np.float64,
      np.uint8,
      np.int8,
      np.int32,
      np.int64,
      np.bool_,
  ]
  assert (
      ty in SUPPORTED
  ), f"Only numpy arrays with dtypes in {SUPPORTED} are supported, but got {ty}"


memref_type_to_np_dtype = {
    "mrf16": np.float16,
    "mrf32": np.float32,
    "mrf64": np.float64,
    "mri1": np.bool_,
    "mri8": np.int8,
    "mri32": np.int32,
    "mri64": np.int64,
}
elemental_type_to_ctype = {
    "i1": ctypes.c_bool,
    "i8": ctypes.c_byte,
    "i64": ctypes.c_int,
    "f32": ctypes.c_float,
    "f64": ctypes.c_double,
}

CONSUME_RETURN_FUNC_PREFIX = "refbackend_consume_func_return_"


def get_return_funcs(module):
  return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
  return_funcs = []
  with module.context:
    for func in module.body:
      # Returns strings of the form `"refbackend.."` so `"` is deleted.
      func_name = str(func.attributes["sym_name"]).replace('"', "")
      if func_name[:return_prefix_len] == CONSUME_RETURN_FUNC_PREFIX:
        return_funcs.append(func_name)

  return return_funcs


def get_ctype_func(func_name):
  return_prefix_len = len(CONSUME_RETURN_FUNC_PREFIX)
  ret_types = func_name[return_prefix_len:].split("_")
  ctypes_arg = [None]
  for type in ret_types:
    if type in elemental_type_to_ctype:
      ctypes_arg.append(elemental_type_to_ctype[type])
    elif type in memref_type_to_np_dtype:
      ctypes_arg.append(ctypes.POINTER(UnrankedMemRefDescriptor))
    else:
      assert False, f"Not supported type: {type}"

  return ctypes.CFUNCTYPE(*ctypes_arg), ret_types


# https://stackoverflow.com/a/68198336/9045206
CData = ctypes._SimpleCData.__mro__[-2]

_MLIR_RUNNER_UTILS_LIB_ENV = "MLIR_RUNNER_UTILS_LIB"
_MLIR_C_RUNNER_UTILS_LIB_ENV = "MLIR_C_RUNNER_UTILS_LIB"
_MLIR_ASYNC_RUNTIME_LIB_ENV = "MLIR_ASYNC_RUNTIME_LIB"


class LLVMJITBackendInvoker:
  return_func: Optional[Callable] = None

  def __init__(self,
               module,
               consume_return_func=None,
               opt_level=2,
               shared_libs=None):
    if shared_libs is None:
      shared_libs = [
          os.environ[_MLIR_RUNNER_UTILS_LIB_ENV],
          os.environ[_MLIR_C_RUNNER_UTILS_LIB_ENV],
          os.environ[_MLIR_ASYNC_RUNTIME_LIB_ENV],
      ]
    self.ee = ExecutionEngine(module,
                              opt_level=opt_level,
                              shared_libs=shared_libs)
    if consume_return_func is not None:
      return_funcs = get_return_funcs(module)
      assert len(return_funcs) == 1, f"multiple return funcs not supported"
      self.return_func = return_funcs[0]
      ctype_wrapper, ret_types = get_ctype_func(self.return_func)
      self.ret_types = ret_types
      self.ee.register_runtime(self.return_func,
                               ctype_wrapper(consume_return_func))

  def __getattr__(self, function_name: str):
    _get = super().__getattribute__

    def invoke(*args):
      ffi_args = []
      for arg in args:
        if isinstance(arg, CData):
          ffi_args.append(arg)
        else:
          assert_arg_type_is_supported(arg.dtype)
          ffi_args.append(
              ctypes.pointer(
                  # TODO(max): this is a hack to handle refbackend
                  # in principle this has nothing to do with anything
                  # refbackend related
                  ctypes.pointer(get_unranked_memref_descriptor(arg))
                  if _get("return_func") is not None else ctypes.
                  pointer(get_ranked_memref_descriptor(arg))))

      self.ee.invoke(function_name, *ffi_args)

    return invoke


class LLVMJITBackend:

  def __init__(
      self,
      shared_libs: Optional[list[str]] = None,
  ):
    if shared_libs is None:
      shared_libs = [
          os.environ[_MLIR_C_RUNNER_UTILS_LIB_ENV],
          os.environ[_MLIR_RUNNER_UTILS_LIB_ENV],
      ]
    self.shared_libs = shared_libs

  def compile(
      self,
      module: Module,
      pipeline: str,
      kernel_name="main",
      enable_ir_printing=False,
  ):

    def cb(op):
      try:
        return kernel_name == op.sym_name.value
      except:
        return False

    needs_cface = "to-llvm" in pipeline
    pipeline_str = pipeline

    if needs_cface:
      kernel_func = find_ops(module.operation, cb)
      assert len(kernel_func) == 1, f"kernel func {kernel_func} not found"
      kernel_func[0].attributes["llvm.emit_c_interface"] = UnitAttr.get()

    return run_pipeline(
        module,
        pipeline=pipeline_str,
        description="Lowering IR",
        enable_ir_printing=enable_ir_printing,
    )

  def load(self,
           module,
           consume_return_func=None,
           opt_level=2) -> LLVMJITBackendInvoker:
    return LLVMJITBackendInvoker(
        module,
        opt_level=opt_level,
        shared_libs=self.shared_libs,
        consume_return_func=consume_return_func,
    )
