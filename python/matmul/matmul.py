import sys, time

from collections.abc import Callable
from typing import Any, Optional, Sequence, Type

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.utils import *


################################################################################
### Matmul
################################################################################
#   Op def: (     m,     n,     k )
#    Iters: ({Par(), Par(), Red()})
#               A       B       C
#   Layout: {{m, k}, {k, n}, {m, n}}
def matmul_shapes_builder(M: int, N: int, K: int):
  return [[M, K], [K, N], [M, N]]


def matmul_gflop_count_builder(M: int, N: int, K: int):
  return float(2.0 * M * N * K) / float(1e9)


def matmul_types_mlir_builder(M: int, N: int, K: int,
                              compiled_function_element_types: Sequence[Type]):
  if len(compiled_function_element_types) != 3:
    raise Exception(
        f'expected 3 element types got: {compiled_function_element_types}')

  shapes = matmul_shapes_builder(M, N, K)
  if len(shapes) != 3:
    raise Exception(f'expected 3 shapes got: {shapes}')

  return [RankedTensorType.get(s, t) for s, t in \
       zip(shapes, compiled_function_element_types)]


def matmul_tensors_np_builder(M: int, N: int, K: int,
                              np_types: Sequence[np.dtype]):
  shapes = matmul_shapes_builder(M, N, K)
  if len(shapes) != 3:
    raise Exception(f'expected 3 shapes got: {shapes}')

  tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
  tensors[len(tensors) - 1].fill(0.)
  return tensors


# TODO: Orthogonal configuration object.
avx512 = True


def build_matmul_under_context_manager(name: str, types: Sequence[Type]):
  global avx512

  if len(types) != 3:
    raise Exception(f'Expected 3 types, got: {types}')

  # Actual benchmarked function called under entry_point_name.
  func = builtin.FuncOp(name, (types, [types[2]]))
  # TODO: need something much more flexible to add func argument attributes.
  attach_inplaceable_attributes(func, rank=2, inplaceable=[False, False, True])
  attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

  acc_type = types[2].element_type
  with InsertionPoint(func.add_entry_block()):
    zero = arith.ConstantOp(acc_type, 0.0)
    tensor_zero = linalg.FillOp(output=func.arguments[2], value=zero)
    matmul = linalg.matmul(
        func.arguments[0], func.arguments[1], outs=[tensor_zero])
    # linalg.matmul returns a Value instead of OpView, so we have to manually
    # wrap it in a list here.
    std.ReturnOp([matmul])

  return func


def matmul_check(A: np.dtype, B: np.dtype, C: np.dtype):
  if not np.allclose(C, np.dot(A, B)):
    delta = C - np.dot(A, B)
    max_abs_delta = max(delta.max(), delta.min(), key=abs)
    raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')
