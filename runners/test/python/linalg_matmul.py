# RUN: %PYTHON %s 2>&1 | FileCheck %s

# Bootstrap our local extensions first.
# TODO: Requires that both ${LLVM_INSTALL}/python and ./build are on
# PYTHONPATH
import runners

import sys, time
from collections.abc import Callable

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
// Activat manually for AVX-512.
   passthrough = [["target-cpu", "skylake-avx512"], ["prefer-vector-width", "512"]],
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


def fuse(module: Module,
         func_name: str,
         op_name: str,
         tile_sizes: list,
         pad=False):
  pad_str = f'fuse-padding' if pad else ''
  tile_str = f'tile-sizes={",".join([str(ts) for ts in tile_sizes])}'
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     fuse '
              f'     {pad_str}'
              f'     {tile_str}}}),'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def tile_and_pad(module: Module,
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
              f'     {hoist_padding_str}}}),'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def vectorize(module: Module, func_name: str, op_name: str):
  pipeline = (f'func(linalg-tensor-codegen-strategy{{anchor-func={func_name} '
              f'     anchor-op={op_name} '
              f'     vectorize '
              f'     vectorize-padding}}),'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def lower_to_llvm(module: Module):
  pipeline = (f'linalg-comprehensive-bufferize-inplace,'
              f'func(convert-linalg-to-loops,'
              f'     convert-vector-to-scf{{full-unroll=true}}),'
              f'canonicalize,'
              f'cse,'
              f'lower-affine,'
              f'convert-scf-to-std,'
              f'convert-vector-to-llvm,'
              f'convert-std-to-llvm')
  PassManager.parse(pipeline).run(module)


def bufferize(module: Module):
  pipeline = (f'linalg-comprehensive-bufferize-inplace,'
              f'canonicalize,'
              f'cse')
  PassManager.parse(pipeline).run(module)


def pre_transform(module, boilerplate_code):
  import mlir.conversions
  import mlir.dialects.linalg.passes
  import mlir.transforms

  # TODO: Allow cloning functions from one module to another.
  # Atm we have to resort to string concatenation.
  module = Module.parse(
      str(module.operation.regions[0].blocks[0].operations[0].operation) +
      boilerplate_code)

  return module


def expert_compilerr_1(module, boilerplate_code):
  module = pre_transform(module, boilerplate_code)
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256, 256])
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [64, 64, 64])
  tile_and_pad(
      module,
      'matmul_on_tensors',
      'linalg.matmul', [8, 16, 32],
      pad=True,
      hoist_padding=2)
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  bufferize(module)
  lower_to_llvm(module)
  return module


def expert_compilerr_2(module, boilerplate_code):
  module = pre_transform(module, boilerplate_code)
  fuse(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256])
  fuse(module, 'matmul_on_tensors', 'linalg.matmul', [8, 16])
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.matmul', [0, 0, 32])
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  vectorize(module, 'matmul_on_tensors', 'linalg.fill')
  bufferize(module)
  lower_to_llvm(module)
  return module


def expert_compilerr_3(module, boilerplate_code):
  module = pre_transform(module, boilerplate_code)
  fuse(module, 'matmul_on_tensors', 'linalg.matmul', [256, 256])
  tile_and_pad(
      module,
      'matmul_on_tensors',
      'linalg.matmul', [8, 16, 32],
      pad=True,
      hoist_padding=3)
  vectorize(module, 'matmul_on_tensors', 'linalg.matmul')
  tile_and_pad(module, 'matmul_on_tensors', 'linalg.fill', [8, 32])
  vectorize(module, 'matmul_on_tensors', 'linalg.fill')
  bufferize(module)
  lower_to_llvm(module)
  return module


# Counts FMA as 2 ops.
def gflop_count(M: int, N: int, K: int):
  return (2.0 * M * N * K) / 1e9


def build_matmul_under_context_manager(M: int, N: int, K: int, ITERS, np_type,
                                       transform):
  # Only f32 supported for now until we can build the whole MLIR snippet from python.
  elem_type = F32Type.get() if np_type is np.float32 else None
  assert (elem_type != None)

  # Build module and function to benchmark.
  module = Module.create()
  with InsertionPoint(module.body):

    @builtin.FuncOp.from_py_func(
        RankedTensorType.get((M, K), elem_type),
        RankedTensorType.get((K, N), elem_type),
        RankedTensorType.get((M, N), elem_type))
    def matmul_on_tensors(lhs, rhs, out):
      # TODO: in the future, should be writeable more concisely as:
      #   zero = std.constant(0.0, elem_type)
      #   tmp = linalg.fill(out, zero)
      #   linalg.matmul(lhs, rhs, tmp)
      zero = std.ConstantOp(
          value=FloatAttr.get(elem_type, 0.), result=elem_type).result
      tensor_zero = linalg.FillOp(output=out, value=zero).results[0]
      return linalg.matmul(lhs, rhs, outs=[tensor_zero])

  # JIT compile.
  start = time.time()
  execution_engine = ExecutionEngine(
      transform(module, boilerplate(M=M, N=N, K=K)))
  elapsed_compilation_s = time.time() - start
  print(f'compilation in {elapsed_compilation_s:.{4}}s')

  return execution_engine


def compile_and_test_linalg_matmul(M: int, N: int, K: int, ITERS: int,
                                   np_type: np.dtype, transform: Callable):
  A = np.random.rand(M, K).astype(np_type)
  B = np.random.rand(K, N).astype(np_type)
  C = np.random.rand(M, N).astype(np_type)
  C.fill(0.)

  # Arguments must be passed as pointers.
  A_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(A)))
  B_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(B)))
  C_memref_ptr = ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(C)))
  index_ptr_t = ctypes.c_longlong * 1

  with Context() as ctx, Location.unknown():
    # Compile.
    execution_engine = build_matmul_under_context_manager(
        M, N, K, ITERS, np_type, transform)

    # Dry-run.
    start = time.time()
    n_iters_dry_run = 1
    execution_engine.invoke('main', A_memref_ptr, B_memref_ptr, C_memref_ptr,
                            index_ptr_t(n_iters_dry_run))
    elapsed_dry_run_s_per_iteration = (time.time() - start) / n_iters_dry_run
    gflop_per_s_per_dry_run_iteration = gflop_count(M, N, K) / (
        elapsed_dry_run_s_per_iteration)
    print(f'dry_run in {elapsed_dry_run_s_per_iteration:.{4}}s per iteration '
          f'sec ({gflop_per_s_per_dry_run_iteration:.{4}} GFlop/s) ')

    # Run for ITERS and report timing.
    start = time.time()
    execution_engine.invoke('main', A_memref_ptr, B_memref_ptr, C_memref_ptr,
                            index_ptr_t(ITERS))
    elapsed_s_per_iteration = (time.time() - start) / ITERS
    gflop_per_s_per_iteration = gflop_count(M, N, K) / (elapsed_s_per_iteration)

    # Check results vs NP and print timings.
    delta = C - np.dot(A, B)
    success = 'SUCCESS' if np.allclose(C, np.dot(A, B)) else 'FAILURE'
    max_abs_delta = max(delta.max(), delta.min(), key=abs)
    print(f'max_abs_delta: {max_abs_delta} -> {success} '
          f'in {elapsed_s_per_iteration:.{4}}s per iteration '
          f'sec ({gflop_per_s_per_iteration:.{4}} GFlop/s) ')


def test_numpy_matmul(M: int, N: int, K: int, ITERS, np_type):
  A = np.random.rand(M, K).astype(np_type)
  B = np.random.rand(K, N).astype(np_type)
  C = np.random.rand(M, N).astype(np_type)
  C.fill(0.)

  # Untimed dry-run iteration.
  start = time.time()
  n_iters_dry_run = 1
  np.dot(A, B, out=C)
  elapsed_dry_run_s_per_iteration = (time.time() - start) / n_iters_dry_run
  gflop_per_s_per_dry_run_iteration = gflop_count(M, N, K) / (
      elapsed_dry_run_s_per_iteration)
  print(f'xxxxxxxxxx : numpy dry_run time on {1} threads '
        f'in {elapsed_dry_run_s_per_iteration:.{4}}s per iteration '
        f'sec ({gflop_per_s_per_dry_run_iteration:.{4}} GFlop/s) ')

  elapsed_s = 0.
  for iters in range(ITERS):
    C.fill(0.)
    start = time.time()
    np.dot(A, B, out=C)
    elapsed_s = elapsed_s + (time.time() - start)
  elapsed_s_per_iteration = elapsed_s / ITERS
  gflop_per_s_per_iteration = gflop_count(M, N, K) / (elapsed_s_per_iteration)
  print(f'xxxxxxxxxx : numpy time on {1} threads '
        f'in {elapsed_s_per_iteration:.{4}}s per iteration '
        f'sec ({gflop_per_s_per_iteration:.{4}} GFlop/s) ')


def test_torch_matmul(M: int, N: int, K: int, ITERS, np_type):
  import torch
  torch.set_num_threads(1)
  A = torch.rand(M, K)
  B = torch.rand(K, N)
  C = torch.rand(M, N)
  C.fill_(0.)

  # Untimed dry-run iteration.
  start = time.time()
  n_iters_dry_run = 1
  torch.mm(A, B, out=C)
  elapsed_dry_run_s_per_iteration = (time.time() - start) / n_iters_dry_run
  gflop_per_s_per_dry_run_iteration = gflop_count(M, N, K) / (
      elapsed_dry_run_s_per_iteration)
  print(f'xxxxxxxxxx : torch dry_run time on {torch.get_num_threads()} threads '
        f'in {elapsed_dry_run_s_per_iteration:.{4}}s per iteration '
        f'sec ({gflop_per_s_per_dry_run_iteration:.{4}} GFlop/s) ')

  elapsed_s = 0.
  for iters in range(ITERS):
    C.fill_(0.)
    start = time.time()
    torch.mm(A, B, out=C)
    elapsed_s = elapsed_s + (time.time() - start)
  elapsed_s_per_iteration = elapsed_s / ITERS
  gflop_per_s_per_iteration = gflop_count(M, N, K) / (elapsed_s_per_iteration)
  print(f'xxxxxxxxxx : torch time on {torch.get_num_threads()} threads '
        f'in {elapsed_s_per_iteration:.{4}}s per iteration '
        f'sec ({gflop_per_s_per_iteration:.{4}} GFlop/s) ')


# CHECK-NOT: FAILURE
n_iters = 10
benchmark_torch = False
problem_size_list = [[128, 192, 256], [256, 256, 256], [1024, 1024, 1024]]
for np_type in [np.float32]:
  for problem_sizes in problem_size_list:
    M, N, K = problem_sizes
    # Init printing.
    print(f'\n###############################################################\n'
          f'Problem size {M}x{N}x{K} ({gflop_count(M, N, K):.{4}} GFlops)')
    for expert in [expert_compilerr_1, expert_compilerr_2, expert_compilerr_3]:
      compile_and_test_linalg_matmul(M, N, K, n_iters, np_type, expert)
    # For single-threaded apples-to-apples comparisons, run with:
    # MKL_NUM_THREADS=1 ATEN_NUM_THREADS=1 OMP_NUM_THREADS=1 TBB_NUM_THREADS=1
    import os
    if os.environ.get('BENCHMARK_NUMPY'):
      test_numpy_matmul(M, N, K, n_iters, np_type)
    if os.environ.get('BENCHMARK_TORCH'):
      test_torch_matmul(M, N, K, n_iters, np_type)
