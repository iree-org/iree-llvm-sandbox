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