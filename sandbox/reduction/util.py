import sys, time
from collections.abc import Callable

import numpy as np
import torch

from mlir.ir import *
from mlir.dialects import builtin
from mlir.dialects import linalg
from mlir.dialects import scf
from mlir.dialects import std
from mlir.execution_engine import *
from mlir.runtime import *
from mlir.dialects.linalg.opdsl.lang import *

from typing import Sequence, Optional

from ..core.harness import *
from ..core.experts import *
from ..core.compilation import f32, attach_inplaceable_attributes, attach_passthrough, emit_benchmarking_function

avx512 = True


def gflop_count_reduction(M: int, N: int):
  return (M * N) / 1e9


def np_type_to_mlir_type(np_type: np.dtype):
  if np_type == np.float16:
    return F16Type.get()
  elif np_type == np.float32:
    return F32Type.get()
  elif np_type == np.float64:
    return F64Type.get()
  else:
    raise Exception(f'unknown scalar type: {scalar_type}')


def setup_reduction_np(M: int, N: int, np_type: np.dtype):
  A = np.random.rand(M, N).astype(np_type)
  B = np.random.rand(M).astype(np_type)
  B.fill(0.)
  return [A, B]


def get_reduction_types(M: int, N: int, input_elem_type, output_elem_type):
  input_tensor_type = RankedTensorType.get([M, N], input_elem_type)
  output_tensor_type = RankedTensorType.get([M], output_elem_type)
  return input_tensor_type, output_tensor_type


@linalg_structured_op
def column_reduction(
    A=TensorDef(T, S.M, S.N), B=TensorDef(T, S.M, output=True)):
  domain(D.m, D.n)
  B[D.m] += A[D.m, D.n]

def emit_compute_function(name: str, types: Sequence[Type]):
  # Actual benchmarked function called under entry_point.
  func = builtin.FuncOp(name, (types, [types[1]]))
  attach_inplaceable_attributes(func, rank=2, inplaceable=[False, True])
  global avx512
  attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)
  # ArrayAttr.get([StringAttr.get('alignstack'),
  #                StringAttr.get('4')])

  output_elem_type = types[1].element_type
  with InsertionPoint(func.add_entry_block()):
    zero = std.ConstantOp(output_elem_type, 0.0)
    tensor_zero = linalg.FillOp(output=func.arguments[1], value=zero)

    result = column_reduction(func.arguments[0], outs=[tensor_zero])
    # linalg.generic returns a Value instead of OpView, so we have to manually
    # wrap it in a list here.
    std.ReturnOp([result])

  return func


# The `reduction_main` function entry point connects MLIR compiled files to python
# allocated tensors. This encodes the runtime / compiler contract that:
#   1. The memory corresponding to the `%B : !output_tensor_t` can be safely
#      written by the compiled code (i.e. linalg.inplaceable = true`).
#   2. The assumed memory layout is the canonical (i, j) order.
# This represents the minimal contract to connect external and compiled code to
# properly test the e2e compilation chain with all batteries included.
#
# For more advanced use cases, including considerations related to parallel
# runtime tasks and special allocators, a runtime abstraction and a more robust
# contract are needed. This is orthogonal to evaluating and benchmarking codegen
# and is the responsibility of projects such as IREE and TFRT.


def build_reduction_under_context_manager(entry_point: str,
                                          fun_to_benchmark: str,
                                          transform: Callable, M: int, N: int,
                                          input_elem_type, output_elem_type):
  global avx512

  # Build module and function to benchmark.
  module = Module.create()
  with InsertionPoint(module.body):
    types = get_reduction_types(M, N, input_elem_type, output_elem_type)
    func = emit_compute_function(fun_to_benchmark, types)
    wrapper = emit_benchmarking_function(entry_point, func)
    attach_passthrough(wrapper, avx512=avx512)

  # JIT compile.
  start = time.time()
  transformed_module = transform(entry_point, module)
  execution_engine = ExecutionEngine(transformed_module)
  elapsed_compilation_s = time.time() - start
  print(f'compilation in {elapsed_compilation_s:.{4}}s')

  return module, execution_engine


def compile_and_test_linalg_reduction(M: int,
                                      N: int,
                                      ITERS: int,
                                      np_type: np.dtype,
                                      transform: Callable,
                                      dry_run: bool = True):
  entry_point = 'reduction_main'
  fun_to_benchmark = 'reduction_on_tensors'

  # np's A and B are hoisted out so they aren't garbage collected.
  A, B = setup_reduction_np(M, N, np_type)

  def setup_fun():
    # Arguments must be passed as pointers.
    A_memref_ptr, B_memref_ptr = (
        ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(t)))
        for t in (A, B))
    return A_memref_ptr, B_memref_ptr

  def compile_fun(A_memref_ptr, B_memref_ptr):
    with Context() as ctx, Location.unknown():
      module, execution_engine = build_reduction_under_context_manager(
          entry_point,
          fun_to_benchmark,
          transform,
          M=M,
          N=N,
          input_elem_type=np_type_to_mlir_type(np_type),
          output_elem_type=np_type_to_mlir_type(np_type))
      return module, execution_engine

  def run_fun(A_memref_ptr, B_memref_ptr, **kwargs):
    index_ptr_t = ctypes.c_longlong * 1
    kwargs['execution_engine'].invoke(entry_point, A_memref_ptr, B_memref_ptr,
                                      index_ptr_t(kwargs['n_iters']))

  # Check results vs NP and print timings.
  # Note that MLIR directly modifies np's tensor memory and the memref_ptr
  # operands are unused here: we can directly look at the result in C.
  def check_fun(A_memref_ptr, B_memref_ptr):
    if not np.allclose(B, np.sum(A, axis=1)):
      delta = B - np.dot(A)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  setup_and_invoke(
      setup_fun,
      run_fun,
      ITERS,
      gflop_count_reduction(M, N),
      compile_fun=compile_fun,
      check_fun=check_fun)


def test_numpy_reduction(M: int, N: int, ITERS: int, np_type):

  def setup_fun():
    return setup_reduction_np(M, N, np_type)

  def run_fun(A, B, **kwargs):
    for _ in range(kwargs['n_iters']):
      B.fill(0.)
      np.sum(A, axis=1, out=B)

  setup_and_invoke(setup_fun, run_fun, ITERS, gflop_count_reduction(M, N))


def test_torch_reduction(M: int, N: int, ITERS: int, np_type, num_threads=2):

  def setup_fun():
    torch.set_num_threads(num_threads)
    return [torch.from_numpy(t) for t in setup_reduction_np(M, N, np_type)]

  def run_fun(A, B, **kwargs):
    for _ in range(kwargs['n_iters']):
      B.fill_(0.)
      torch.sum(A, dim=1, out=B)

  setup_and_invoke(setup_fun, run_fun, ITERS, gflop_count_reduction(M, N))
