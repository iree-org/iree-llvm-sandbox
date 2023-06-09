# RUN: %PYTHON %s | FileCheck %s

import typing

import torch
import numpy as np

import mlir_structured
from mlir_structured.triton.jit import jit
import triton
import triton.language as tl


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# Copied from mlir.sandbox.utils. That package uses the vanilla `mlir` package
# instead of `mlir_structured` as the rest of this file, so they are
# incompatible.
def realign(allocated_unaligned: np.ndarray, byte_alignment: int = 64):
  shape = allocated_unaligned.shape
  dt = allocated_unaligned.dtype
  effective_size_in_bytes = np.prod(shape) * np.dtype(dt).itemsize
  total_size_in_bytes = effective_size_in_bytes + byte_alignment
  buf = np.empty(total_size_in_bytes, dtype=np.byte)
  off = (-buf.ctypes.data % byte_alignment)
  allocated_aligned = buf[off:off +
                          effective_size_in_bytes].view(dt).reshape(shape)
  np.copyto(allocated_aligned, allocated_unaligned)
  assert allocated_aligned.ctypes.data % byte_alignment == 0
  return allocated_aligned


def create_aligned_tensor(data: typing.Any,
                          dtype: torch.types._dtype | None = None
                         ) -> torch.Tensor:
  np_dtype = torch.tensor(data[0], dtype=dtype).numpy().dtype
  np_array = np.array(data, dtype=np_dtype)
  np_array = realign(np_array)
  return torch.as_tensor(np_array, dtype=dtype)


# CHECK-LABEL: TEST: addptr_scalar
@run
def addptr_scalar():

  @jit
  def kernel(ptr):
    x = tl.load(ptr)
    tl.store(ptr + 1, x)

  X = create_aligned_tensor([42, 0], dtype=torch.int32)
  kernel[(1,)](X)

  # CHECK-NEXT: tensor([42, 42], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: load_store_scalar
@run
def load_store_scalar():

  @jit
  def kernel(ptr, i, CONST: tl.constexpr):
    x = tl.load(ptr)
    x += i + CONST
    tl.store(ptr, x)

  X = create_aligned_tensor([20], dtype=torch.int32)
  kernel[(1,)](X, 21, CONST=1)

  # CHECK-NEXT: tensor([42], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: program_id
@run
def program_id():

  @jit
  def kernel(ptr):
    x = tl.program_id(axis=0)
    y = tl.program_id(axis=1)
    z = tl.program_id(axis=2)
    tl.store(ptr + x * 4 + y * 2 + z, x * 4 + y * 2 + z)

  X = create_aligned_tensor([42] * 8, dtype=torch.int32)
  kernel[(2, 2, 2)](X)

  # CHECK-NEXT: tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: num_programs
@run
def num_programs():

  @jit
  def kernel(ptr):
    n_x = tl.num_programs(axis=0)
    n_y = tl.num_programs(axis=1)
    n_z = tl.num_programs(axis=2)
    tl.store(ptr + 0, n_x)
    tl.store(ptr + 1, n_y)
    tl.store(ptr + 2, n_z)

  X = create_aligned_tensor([42] * 3, dtype=torch.int32)
  kernel[(1, 2, 3)](X)

  # CHECK-NEXT: tensor([1, 2, 3], dtype=torch.int32)
  print(X)
