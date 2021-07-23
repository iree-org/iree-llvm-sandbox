# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Bootstrap our local extensions first.
# TODO: Come up with a way to make this auto-load.
import mlir.iree_sandbox

import sys
import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *
from mlir.runtime import *


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


# Define a callback_anked function that takes an unranked memref and dumps
# metadata.
@ctypes.CFUNCTYPE(
    None,
    ctypes.POINTER(
        make_nd_memref_descriptor(2, np.ctypeslib.as_ctypes_type(np.float32))),
)
def callback_ranked(ranked_memref):
  descriptor_t = make_nd_memref_descriptor(
      2, np.ctypeslib.as_ctypes_type(np.float32))
  val = ctypes.cast(ranked_memref, ctypes.POINTER(descriptor_t))
  print(f'Inside Ranked Callback:offset={val[0].offset}, '
        f'shape={np.ctypeslib.as_array(val[0].shape)}, '
        f'strides={np.ctypeslib.as_array(val[0].strides)}, ')
  arr = ranked_memref_to_numpy(ranked_memref)
  print(f'Inside Ranked Callback:np.shape={arr.shape}, '
        f'np.strides={arr.strides}, ')


# Define a callback_unranked function that takes an unranked memref and dumps
# metadata.
@ctypes.CFUNCTYPE(None, ctypes.POINTER(UnrankedMemRefDescriptor))
def callback_unranked(unranked_memref):
  descriptor_t = make_nd_memref_descriptor(
      unranked_memref[0].rank, np.ctypeslib.as_ctypes_type(np.float32))
  val = ctypes.cast(unranked_memref[0].descriptor, ctypes.POINTER(descriptor_t))
  print(f'Inside Unranked Callback:offset={val[0].offset}, '
        f'shape={np.ctypeslib.as_array(val[0].shape)}, '
        f'strides={np.ctypeslib.as_array(val[0].strides)}, ')
  arr = unranked_memref_to_numpy(unranked_memref, np.float32)
  print(f'Inside Unranked Callback:np.shape={arr.shape}, '
        f'np.strides={arr.strides}, ')


def boilerplate(M: int, N: int):
  return f"""
func @callback_ranked_tensor(%arg0: tensor<{M}x{N}xf32>) attributes {{ llvm.emit_c_interface }} {{
  call @some_callback_ranked_into_python(%arg0) : (tensor<{M}x{N}xf32>) -> ()
  return
}}
func private @some_callback_ranked_into_python(tensor<{M}x{N}xf32>) -> () attributes {{ llvm.emit_c_interface }}

func @callback_unranked_tensor(%arg0: tensor<*xf32>) attributes {{ llvm.emit_c_interface }} {{
  call @some_callback_unranked_into_python(%arg0) : (tensor<*xf32>) -> ()
  return
}}
func private @some_callback_unranked_into_python(tensor<*xf32>) -> () attributes {{ llvm.emit_c_interface }}

func @main(%A : tensor<{M}x{N}xf32>) -> () attributes {{ llvm.emit_c_interface }}
{{
  call @callback_ranked_tensor(%A) : (tensor<{M}x{N}xf32>) -> ()

  %a = tensor.cast %A : tensor<{M}x{N}xf32> to tensor<*xf32>
  call @callback_unranked_tensor(%a) : (tensor<*xf32>) -> ()

  return
}}
"""


def bufferize_to_llvm(module: Module, func_name: str, op_name: str):
  pipeline = (f'linalg-comprehensive-bufferize-inplace,'
              f'convert-scf-to-std,'
              f'convert-std-to-llvm')
  PassManager.parse(pipeline).run(module)


def transform(boilerplate_code):
  import mlir.conversions
  import mlir.transforms

  module = Module.parse(boilerplate_code)
  bufferize_to_llvm(module, 'matmul_on_tensors', 'linalg.matmul')

  return module


def test_memref_metadata(M: int, N: int):
  with Context() as ctx, Location.unknown():
    module = Module.create()
    np_type = np.float32
    f32 = F32Type.get()
    execution_engine = ExecutionEngine(transform(boilerplate(M=M, N=N)))

    A = np.random.rand(M, N).astype(np_type)

    A_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(A)))

    execution_engine.register_runtime('some_callback_ranked_into_python',
                                      callback_ranked)
    execution_engine.register_runtime('some_callback_unranked_into_python',
                                      callback_unranked)
    execution_engine.invoke('main', A_memref_ptr)


# CHECK: Inside Ranked Callback:offset=0, shape=[4 8], strides=[8 1],
# CHECK: Inside Ranked Callback:np.shape=(4, 8), np.strides=(32, 4),
# CHECK: Inside Unranked Callback:offset=0, shape=[4 8], strides=[8 1],
# CHECK: Inside Unranked Callback:np.shape=(4, 8), np.strides=(32, 4),
test_memref_metadata(4, 8)
# CHECK: Inside Ranked Callback:offset=0, shape=[128 192], strides=[192   1],
# CHECK: Inside Ranked Callback:np.shape=(128, 192), np.strides=(768, 4),
# CHECK: Inside Unranked Callback:offset=0, shape=[128 192], strides=[192   1],
# CHECK: Inside Unranked Callback:np.shape=(128, 192), np.strides=(768, 4),
test_memref_metadata(128, 192)
# CHECK: Inside Ranked Callback:offset=0, shape=[1024 1024], strides=[1024    1],
# CHECK: Inside Ranked Callback:np.shape=(1024, 1024), np.strides=(4096, 4),
# CHECK: Inside Unranked Callback:offset=0, shape=[1024 1024], strides=[1024    1],
# CHECK: Inside Unranked Callback:np.shape=(1024, 1024), np.strides=(4096, 4),
test_memref_metadata(1024, 1024)
