import sys, time

from collections.abc import Callable
from typing import Any, Optional, Sequence, Type

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std
from mlir.dialects.linalg.opdsl.lang import *

from ..core.utils import *
from ..core.compilation import attach_inplaceable_attributes, attach_passthrough


################################################################################
### Reduction
################################################################################
#   Op def: (     m,     k )
#    Iters: ({Par(), Red()})
#             I       O
#   Layout: {{m, k}, {m}}
def reduction_2d_shapes_builder(M: int, K: int):
  return [[M, K], [M]]


def reduction_2d_gflop_count_builder(M: int, N: int):
  return float(M * N) / float(1e9)


def reduction_2d_types_mlir_builder(
    M: int,
    K: int,
    # TODO: Sequence[MlirType]
    compiled_function_element_types: Sequence[Any]):
  if len(compiled_function_element_types) != 2:
    raise Exception(
        f'expected 2 element types got: {compiled_function_element_types}')

  shapes = reduction_2d_shapes_builder(M, K)
  if len(shapes) != 2:
    raise Exception(f'expected 2 shapes got: {shapes}')

  return [RankedTensorType.get(s, t) for s, t in \
       zip(shapes, compiled_function_element_types)]


def reduction_2d_tensors_np_builder(
    M: int,
    K: int,
    # TODO: Sequence[NpType]
    np_types: Sequence[Any]):
  shapes = reduction_2d_shapes_builder(M, K)
  if len(shapes) != 2:
    raise Exception(f'expected 2 shapes got: {shapes}')

  tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
  tensors[len(tensors) - 1].fill(0.)
  return tensors




@linalg_structured_op
def column_reduction(
    A=TensorDef(T, S.M, S.N), B=TensorDef(T, S.M, output=True)):
  domain(D.m, D.n)
  B[D.m] += A[D.m, D.n]


# TODO: Orthogonal configuration object.
avx512 = True


def build_reduction_2d_under_context_manager(name: str, types: Sequence[Type]):
  global avx512

  if len(types) != 2:
    raise Exception(f'Expected 2 types, got: {types}')

  # Actual benchmarked function called under entry_point_name.
  func = builtin.FuncOp(name, (types, [types[1]]))
  # TODO: need something much more flexible to add func argument attributes.
  attach_inplaceable_attributes(func, rank=2, inplaceable=[False, True])
  attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

  output_elem_type = types[1].element_type
  with InsertionPoint(func.add_entry_block()):
    zero = arith.ConstantOp(output_elem_type, 0.0)
    tensor_zero = linalg.FillOp(output=func.arguments[1], value=zero)
    result = column_reduction(func.arguments[0], outs=[tensor_zero])
    # linalg.matmul returns a Value instead of OpView, so we have to manually
    # wrap it in a list here.
    std.ReturnOp([result])

  return func


def reduction_2d_check(A, B):
  if not np.allclose(B, np.sum(A, axis=1)):
    delta = B - np.dot(A)
    max_abs_delta = max(delta.max(), delta.min(), key=abs)
    raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')
