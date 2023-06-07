# RUN: %PYTHON %s | FileCheck %s

import torch

import mlir_structured
from mlir_structured.triton.jit import jit
import triton
import triton.language as tl


def run(f):
  print("\nTEST:", f.__name__)
  f()
  return f


# CHECK-LABEL: TEST: addptr_scalar
@run
def addptr_scalar():

  @jit
  def kernel(ptr):
    x = tl.load(ptr)
    tl.store(ptr + 1, x)

  X = torch.tensor([42, 0], dtype=torch.int32)
  kernel[(1,)](X)

  # CHECK-NEXT: tensor([42, 42], dtype=torch.int32)
  print(X)


@jit
def times_two(x):
  return x + x


# CHECK-LABEL: TEST: call_other_kernel
@run
def call_other_kernel():

  @jit
  def kernel(ptr):
    x = tl.load(ptr)
    x = times_two(x)
    tl.store(ptr, x)

  X = torch.tensor([21], dtype=torch.int32)
  kernel[(1,)](X)

  # CHECK-NEXT: tensor([42], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: dot
@run
def dot():

  @jit
  def kernel(ptrA, ptrB, ptrC):
    r = tl.arange(0, 256)
    r = tl.view(r, (16, 16))
    ptrAs = ptrA + r
    ptrBs = ptrB + r
    ptrCs = ptrC + r

    a = tl.load(ptrAs)
    b = tl.load(ptrBs)
    c = tl.dot(a, b)

    tl.store(ptrCs, c)

  A = torch.tensor(list(range(16)) * 16, dtype=torch.float32)
  B = torch.tensor(list(range(4)) * 64, dtype=torch.float32)
  C = torch.tensor([1] * 256, dtype=torch.float32)
  kernel[(1,)](A, B, C)

  # CHECK-NEXT: tensor(46080.)
  print(torch.sum(C))

  # CHECK-NEXT: tensor(46080.)
  torch.matmul(A.reshape([16, 16]),
               B.reshape([16, 16]),
               out=C.reshape([16, 16]))
  print(torch.sum(C))


# CHECK-LABEL: TEST: load_store_scalar
@run
def load_store_scalar():

  @jit
  def kernel(ptr, i, CONST: tl.constexpr):
    x = tl.load(ptr)
    x += i + CONST
    tl.store(ptr, x)

  X = torch.tensor([20], dtype=torch.int32)
  kernel[(1,)](X, 21, CONST=1)

  # CHECK-NEXT: tensor([42], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: load_store_tensor
@run
def load_store_tensor():

  @jit
  def kernel(ptr):
    r = tl.arange(0, 4)
    ptr_r = ptr + r
    t = tl.load(ptr_r)
    t = t + t
    tl.store(ptr_r, t)

  X = torch.tensor(list(range(100, 104)), dtype=torch.int32)
  kernel[(1,)](X)

  # CHECK-NEXT: tensor([200, 202, 204, 206], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: load_store_kd_tensor
@run
def load_store_kd_tensor():

  @jit
  def kernel(ptr):
    r = tl.arange(0, 4)
    r = tl.view(r, (2, 2))
    ptr_r = ptr + r

    t = tl.load(ptr_r)
    t = t + t

    tl.store(ptr_r, t)

  X = torch.tensor(list(range(100, 104)), dtype=torch.int32)
  kernel[(1,)](X)

  # CHECK-NEXT: tensor([200, 202, 204, 206], dtype=torch.int32)
  print(X)


@jit
def get_grid_coords(dummy):
  # Call tl.program_id inside of function to also test tt.call op.
  x = tl.program_id(axis=0)
  y = tl.program_id(axis=1)
  z = tl.program_id(axis=2)
  return (x, y, z)


# CHECK-LABEL: TEST: program_id
@run
def program_id():

  @jit
  def kernel(ptr):
    x, y, z = get_grid_coords(0)
    tl.store(ptr + x * 4 + y * 2 + z, x * 4 + y * 2 + z)

  X = torch.tensor([42] * 8, dtype=torch.int32)
  kernel[(2, 2, 2)](X)

  # CHECK-NEXT: tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32)
  print(X)


# CHECK-LABEL: TEST: view_op
@run
def view_op():

  @jit
  def kernel(ptr):
    r1x4 = tl.arange(0, 4)
    r2x2 = tl.view(r1x4, (2, 2))
    ptr_r2x2 = ptr + r2x2
    ptr_r1x4 = tl.view(ptr_r2x2, (4,))
    tl.store(ptr_r1x4, r1x4)

  X = torch.tensor(list(range(100, 104)), dtype=torch.int32)
  kernel[(1,)](X)

  # CHECK-NEXT: tensor([0, 1, 2, 3], dtype=torch.int32)
  print(X)
