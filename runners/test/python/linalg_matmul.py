# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH
import runners

import sys
from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.passmanager import *
from mlir.execution_engine import *


# Log everything to stderr and flush so that we have a unified stream to match
# errors/info emitted by MLIR to stderr.
def log(*args):
  print(*args, file=sys.stderr)
  sys.stderr.flush()


def boilerplate(M: int, N: int, K: int, ITERS: int):
  return f"""
func @main() -> f32 attributes {{llvm.emit_c_interface}} {{
  %v0 = constant 0.0 : f32
  %v1 = constant 1.0 : f32
  %v2 = constant 2.0 : f32

  %iA = linalg.init_tensor [{M}, {K}] : tensor<{M}x{K}xf32>
  %iB = linalg.init_tensor [{K}, {N}] : tensor<{K}x{N}xf32>
  %iC = linalg.init_tensor [{M}, {N}] : tensor<{M}x{N}xf32>
  %A = linalg.fill(%iA, %v1) : tensor<{M}x{K}xf32>, f32 -> tensor<{M}x{K}xf32>
  %B = linalg.fill(%iB, %v2) : tensor<{K}x{N}xf32>, f32 -> tensor<{K}x{N}xf32>
  %C = linalg.fill(%iC, %v0) : tensor<{M}x{N}xf32>, f32 -> tensor<{M}x{N}xf32>

  %c0 = constant 0: index
  %c1 = constant 1: index
  %iters = constant {ITERS}: index
  %res = scf.for %arg0 = %c0 to %iters step %c1 iter_args(%dummy = %C) -> (tensor<{M}x{N}xf32>) {{
    %r = call @matmul_on_tensors(%A, %B, %C) :
      (tensor<{M}x{K}xf32>, tensor<{K}x{N}xf32>, tensor<{M}x{N}xf32>) -> (tensor<{M}x{N}xf32>)
    scf.yield %r : tensor<{M}x{N}xf32>
  }}

  %0 = vector.transfer_read %res[%c0, %c0], %v0 : tensor<{M}x{N}xf32>, vector<2xf32>
  %e0 = vector.extract %0[0] : vector<2xf32>

  // TODO: FFI-based solution to allow testing and printing with python code.
  return %e0 : f32
}}
"""


def tile_and_pad(mod: Module,
                 func_name: str,
                 op_name: str,
                 tile_sizes: list,
                 pad=False,
                 hoist_padding=None):
  pad_str, hoist_padding_str = '', ''
  tile_sizes_str = f'tile-sizes={",".join([str(ts) for ts in tile_sizes])}'
  if pad:
    pad_str = 'pad'
  if hoist_padding:
    hoist_padding_str = f'hoist-padding={hoist_padding}'
  pipeline = (
      f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
      f'anchor-op={op_name} {tile_sizes_str} {pad_str} {hoist_padding_str}}})')
  PassManager.parse(pipeline).run(mod)


def vectorize(mod: Module, func_name: str, op_name: str):
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'anchor-op={op_name} vectorize vectorize-padding}})')
  PassManager.parse(pipeline).run(mod)


def bufferize_to_llvm(mod: Module, func_name: str, op_name: str):
  pipeline = (
      f'linalg-comprehensive-bufferize-inplace,'
      f'func(convert-linalg-to-loops,convert-vector-to-scf{{full-unroll=true}}),'
      f'lower-affine,convert-scf-to-std,convert-vector-to-llvm,convert-std-to-llvm'
  )
  PassManager.parse(pipeline).run(mod)


def transform(module, boilerplate_code):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  mod = Module.parse(
      str(module.operation.regions[0].blocks[0].operations[0].operation) +
      boilerplate_code)

  tile_and_pad(mod, 'matmul_on_tensors', 'linalg.matmul', [256, 256, 256])
  tile_and_pad(mod, 'matmul_on_tensors', 'linalg.matmul', [32, 32, 64])
  tile_and_pad(
      mod,
      'matmul_on_tensors',
      'linalg.matmul', [2, 4, 16],
      pad=True,
      hoist_padding=6)
  vectorize(mod, 'matmul_on_tensors', 'linalg.matmul')
  bufferize_to_llvm(mod, 'matmul_on_tensors', 'linalg.matmul')

  return mod


def test_matmul(M: int, N: int, K: int, ITERS=1):
  with Context() as ctx, Location.unknown():
    module = Module.create()
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
        transform(module, boilerplate(M=M, N=N, K=K, ITERS=ITERS)))

    # TODO: FFI-based solution to allow testing and printing with python code.
    # Prepare arguments: one result f32.
    # Arguments must be passed as pointers.
    c_float_p = ctypes.c_float * 1
    res = c_float_p(-1.)
    execution_engine.invoke('main', res)

    log('RESULT: ', res[0])


# TODO: More iterations once we have a fill to zero
# the results at each iteration
# CHECK: RESULT: 32.0
test_matmul(4, 8, 16, 1)
# CHECK: RESULT: 512.0
test_matmul(256, 256, 256, 1)
# CHECK: RESULT: 2048.0
test_matmul(1024, 1024, 1024, 1)
