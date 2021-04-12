# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH
import runners

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


def boilerplate(M: int, N: int, K: int):
  return f"""
func @main(%A : tensor<{M}x{K}xf32>, %B : tensor<{K}x{N}xf32>, %C : tensor<{M}x{N}xf32>, %iters : index)
  -> tensor<{M}x{N}xf32>
  attributes {{
      llvm.emit_c_interface,
// Manually set up `__writeable_func_buffer_args_attr__` to allow proper inplace
// behavior. This is akin to a user annotation that the compiler understands.
      __writeable_func_buffer_args_attr__ = ["false", "false", "true"] }}
{{
  %c0 = constant 0: index
  %c1 = constant 1: index

  %res = scf.for %arg0 = %c0 to %iters step %c1 iter_args(%iterC = %C) -> (tensor<{M}x{N}xf32>) {{
    %r = call @matmul_on_tensors(%A, %B, %iterC) :
      (tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>, tensor<{M}x{N}xf32>) -> (tensor<{M}x{N}xf32>)
    scf.yield %r : tensor<{M}x{N}xf32>
  }}

  return %res : tensor<{M}x{N}xf32>
}}
"""


def tile_and_pad(mod: Module,
                 func_name: str,
                 op_name: str,
                 tile_sizes: list,
                 pad=False,
                 hoist_padding=None):
  pad_str, hoist_padding_str = '', ''
  tile_str = f'tile-sizes={",".join([str(ts) for ts in tile_sizes])}'
  if pad:
    pad_str = 'pad'
  if hoist_padding:
    hoist_padding_str = f'hoist-padding={hoist_padding}'
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     {tile_str} '
              f'     {pad_str} '
              f'     {hoist_padding_str}}})')
  PassManager.parse(pipeline).run(mod)


def vectorize(mod: Module, func_name: str, op_name: str):
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     vectorize '
              f'     vectorize-padding}})')
  PassManager.parse(pipeline).run(mod)


def bufferize_to_llvm(module: Module, func_name: str, op_name: str):
  pipeline = (f'linalg-comprehensive-bufferize-inplace,'
              f'func(convert-linalg-to-loops,'
              f'     convert-vector-to-scf{{full-unroll=true}}),'
              f'lower-affine,'
              f'convert-scf-to-std,'
              f'convert-vector-to-llvm,'
              f'convert-std-to-llvm')
  PassManager.parse(pipeline).run(module)


def transform(module, boilerplate_code):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  module = Module.parse(
      str(module.operation.regions[0].blocks[0].operations[0].operation) +
      boilerplate_code)

  # TODO: there seem to be some issues with np.ndarrary.strides compatibility as
  # they operate in "number of bytes" whereas memref strides operate in "number
  # of elements".
  # Reenable transformations when resolved.

  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256, 256])
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [32, 32, 64])
  tile_and_pad(
      module,
      'matmul_on_tensors',
      'linalg.matmul', [2, 4, 16],
      pad=True,
      hoist_padding=6)
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  bufferize_to_llvm(module, 'matmul_on_tensors', 'linalg.matmul')

  return module


def test_matmul(M: int, N: int, K: int, ITERS=1):
  with Context() as ctx, Location.unknown():
    module = Module.create()
    np_type = np.float32
    f32 = F32Type.get()
    with InsertionPoint(module.body):

      @builtin.FuncOp.from_py_func(
          RankedTensorType.get((M, K), f32), RankedTensorType.get((K, N), f32),
          RankedTensorType.get((M, N), f32))
      def matmul_on_tensors(lhs, rhs, out):
        # TODO: enable this.
        # tensor_zero = linalg.fill(out, zero)
        return linalg.matmul(lhs, rhs, outs=[out])

    execution_engine = ExecutionEngine(
        transform(module, boilerplate(M=M, N=N, K=K)))

    A = np.random.rand(M, K).astype(np_type)
    B = np.random.rand(K, N).astype(np_type)
    C = np.random.rand(M, N).astype(np_type)
    C.fill(0.)

    # Arguments must be passed as pointers.
    A_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(A)))
    B_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(B)))
    C_memref_ptr = ctypes.pointer(
        ctypes.pointer(get_ranked_memref_descriptor(C)))
    index_ptr_t = ctypes.c_longlong * 1

    execution_engine.invoke('main', A_memref_ptr, B_memref_ptr, C_memref_ptr,
                            index_ptr_t(ITERS))

    delta = C - np.dot(A, B)
    success = np.allclose(C, np.dot(A, B))
    max_abs_delta = max(delta.max(), delta.min(), key=abs)
    print(f'max_abs_delta: {max_abs_delta} -> SUCCESS = {success}')


# TODO: More iterations once we have a fill to zero the results at each
# iteration inside the matmul_on_tensors func.

# CHECK: SUCCESS = True
test_matmul(4, 8, 16, 1)
# CHECK: SUCCESS = True
test_matmul(128, 192, 256, 1)
# CHECK: SUCCESS = True
test_matmul(1024, 1024, 1024, 1)
