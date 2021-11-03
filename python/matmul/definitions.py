import sys, time

from collections.abc import Callable
from typing import Any, List, Optional, Sequence, Type

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = True


################################################################################
### Matmul
################################################################################
#   Op def: (     m,     n,     k )
#    Iters: ({Par(), Par(), Red()})
#               A       B       C
#   Layout: {{m, k}, {k, n}, {m, n}}
class MatmulProblem(ProblemDefinition):
  """ Problem definition for a single fill + matmul problem."""

  def shapes_builder(self, M: int, N: int, K: int) -> List[List[int]]:
    """Shape builder function.

       Given a list of integer dimensions, return the list of lists of shapes
       of the FuncOp operands. The FuncOp is responsible for distinguishing
       between input operands and results.
    """
    return [[M, K], [K, N], [M, N]]

  def gflop_count_builder(self, M: int, N: int, K: int) -> float:
    """GFlop builder function.

       Given a list of integer dimensions, return the number of GFlops computed.
    """
    return float(2.0 * M * N * K) / float(1e9)

  def gbyte_count_builder(self, M: int, N: int, K: int, lhs_np_type: np.dtype,
                          rhs_np_type: np.dtype,
                          acc_np_type: np.dtype) -> float:
    """GByte builder function.

       Given a list of integer dimensions, return the number of GBytes read or
       written.
    """
    return float(M * N * np.dtype(lhs_np_type).itemsize +
                 M * K * np.dtype(rhs_np_type).itemsize +
                 K * N * np.dtype(acc_np_type).itemsize) / float(1e9)

  def tensors_np_builder(self, M: int, N: int, K: int, lhs_np_type: np.dtype,
                         rhs_np_type: np.dtype,
                         acc_np_type: np.dtype) -> List[np.dtype]:
    """NP tensors building function.

       Given a list of integer dimensions followed by per-operand NP elemental
       types, return constructed NP values of shapes given by `shaped_builder`
       and specified elemental types.
    """
    shapes = self.shapes_builder(M, N, K)
    np_types = [lhs_np_type, rhs_np_type, acc_np_type]
    tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, A: np.dtype, B: np.dtype, C: np.dtype) -> None:
    """NP checking function.

       Given a list of NP values, check the precomputed results matches those
       of the expected reference implementation.
    """
    if not np.allclose(C, np.dot(A, B)):
      delta = C - np.dot(A, B)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, M: int, N: int, K: int, lhs_mlir_type: Type,
                         rhs_mlir_type: Type,
                         acc_mlir_type: Type) -> List[Type]:
    """ MLIR types builder.

        Given a list of NP values, check the precomputed results matches those
        of the expected reference implementation.
    """
    compiled_function_element_types = [
        lhs_mlir_type, rhs_mlir_type, acc_mlir_type
    ]
    shapes = self.shapes_builder(M, N, K)
    return [RankedTensorType.get(s, t) for s, t in \
         zip(shapes, compiled_function_element_types)]

  def build_problem_under_context_manager(self, name: str, lhs_mlir_type: Type,
                                          rhs_mlir_type: Type,
                                          acc_mlir_type: Type):
    # TODO: -> FuncOp
    """MLIR problem builder.

       Given a flat list of MLIR types, build and return the MLIR FuncOp that
       implements the desired computation on those types.
    """
    global avx512

    types = [lhs_mlir_type, rhs_mlir_type, acc_mlir_type]

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, [acc_mlir_type]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    acc_type = acc_mlir_type.element_type
    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(acc_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[2], value=zero)
      matmul = linalg.matmul(
          func.arguments[0], func.arguments[1], outs=[tensor_zero])
      # linalg.matmul returns a Value instead of OpView, so we have to manually
      # wrap it in a list here.
      std.ReturnOp([matmul])

    return func
