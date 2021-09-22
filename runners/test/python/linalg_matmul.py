import sys, time
from collections.abc import Callable

import numpy as np

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import std
from mlir.execution_engine import *
from mlir.runtime import *

from harness import *
from experts import *
from compilation import build_op_under_context_manager, compile_and_callback, f32


def gflop_count_matmul(M: int, N: int, K: int):
  return (2.0 * M * N * K) / 1e9


def setup_matmul_np(M: int, N: int, K: int, np_type: np.dtype):
  A = np.random.rand(M, K).astype(np_type)
  B = np.random.rand(K, N).astype(np_type)
  C = np.random.rand(M, N).astype(np_type)
  C.fill(0.)
  return [A, B, C]


def compile_and_test_linalg_matmul(M: int,
                                   N: int,
                                   K: int,
                                   ITERS: int,
                                   np_type: np.dtype,
                                   transform: Callable,
                                   dry_run: bool = True):
  # np's A, B and C are hoisted out so they aren't garbage collected.
  A, B, C = setup_matmul_np(M, N, K, np_type)

  def setup_fun():
    # Arguments must be passed as pointers.
    A_memref_ptr, B_memref_ptr, C_memref_ptr = (
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(t)))
        for t in (A, B, C))
    return A_memref_ptr, B_memref_ptr, C_memref_ptr

  def run_fun(iters, A_memref_ptr, B_memref_ptr, C_memref_ptr):
    with Context() as ctx, Location.unknown():
      index_ptr_t = ctypes.c_longlong * 1
      module, execution_engine = build_op_under_context_manager(
          linalg.matmul, transform, M=M, N=N, K=K, T1=f32, T2=f32, U=f32)
      execution_engine.invoke('main', A_memref_ptr, B_memref_ptr, C_memref_ptr,
                              index_ptr_t(123))

  # Check results vs NP and print timings.
  # Note that MLIR directly modifies np's tensor memory and the memref_ptr
  # operands are unused here: we can directly look at the result in C.
  def check_fun(A_memref_ptr, B_memref_ptr, C_memref_ptr):
    success = 'SUCCESS' if np.allclose(C, np.dot(A, B)) else 'FAILURE'
    if success == 'SUCCESS':
      print(f'{success} ')
    else:
      delta = C - np.dot(A, B)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      print(f'max_abs_delta: {max_abs_delta} -> {success} ')

  setup_and_invoke(
      setup_fun,
      run_fun,
      ITERS,
      gflop_count_matmul(M, N, K),
      check_fun=check_fun)


def test_numpy_matmul(M: int, N: int, K: int, ITERS: int, np_type):

  def setup_fun():
    return setup_matmul_np(M, N, K, np_type)

  def run_fun(iters, A, B, C):
    for iters in range(iters):
      C.fill(0.)
      np.dot(A, B, out=C)

  setup_and_invoke(setup_fun, run_fun, ITERS, gflop_count_matmul(M, N, K))


def test_torch_matmul(M: int,
                      N: int,
                      K: int,
                      ITERS: int,
                      np_type,
                      num_threads=2):

  def setup_fun():
    import torch
    torch.set_num_threads(num_threads)
    return [torch.from_numpy(t) for t in setup_matmul_np(M, N, K, np_type)]

  def run_fun(iters, A, B, C):
    for iters in range(iters):
      C.fill_(0.)
      torch.mm(A, B, out=C)

  setup_and_invoke(setup_fun, run_fun, ITERS, gflop_count_matmul(M, N, K))
