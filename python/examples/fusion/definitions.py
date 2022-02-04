import os, sys, time

from typing import Any, List, Mapping, Optional, Sequence

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, tensor, scf, std
from mlir.dialects.linalg.opdsl.lang import *

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

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the list of lists of shapes of the FuncOp operands. The
    FuncOp is responsible for distinguishing between input operands and results.
    """
    M, N, K = sizes["M"], sizes["N"], sizes["K"]
    return [[M, K], [K, N], [M, N]]

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """GFlop builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the number of GFlops computed.
    """
    M, N, K = sizes["M"], sizes["N"], sizes["K"]
    return float(2.0 * M * N * K) / float(1e9)

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """GByte builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of data types, return the number of GBytes read or
    written.
    """
    M, N, K = sizes["M"], sizes["N"], sizes["K"]
    lhs_np_type, rhs_np_type, acc_np_type = types
    return float(M * N * np.dtype(lhs_np_type).itemsize +
                 M * K * np.dtype(rhs_np_type).itemsize +
                 K * N * np.dtype(acc_np_type).itemsize) / float(1e9)

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """NumPy tensors building function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of NumPy elemental types, return constructed NP values of
    shapes given by `shape_builder` and specified elemental types.
    """
    shapes = self.shapes_builder(sizes)
    tensors = [
        realign(np.random.rand(*s).astype(t), byte_alignment=64)
        for s, t in zip(shapes, types)
    ]
    # Uncomment to simplify debugging.
    # tensors = [
    #     realign(np.arange(1, np.prod(s) + 1).reshape(s).astype(t), \
    #             byte_alignment=64) \
    #     for s, t in zip(shapes, np_types)
    # ]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, A: np.dtype, B: np.dtype, C: np.dtype) -> None:
    """NumPy checking function.

    Given a list of NumPy values, check the precomputed results matches those of
    the expected reference implementation.
    """
    if not np.allclose(C, np.dot(A, B)):
      delta = C - np.dot(A, B)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f"max_abs_delta: {max_abs_delta} -> FAILURE ")

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """MLIR types builder.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of elemental MLIR types, return MLIR tensor types of the
    shape expected by the function.
    """
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in zip(shapes, types)]

  def build_problem_under_context_manager(
      self, name: str, types: Sequence[Type],
      zero_at_each_iteration: bool) -> builtin.FuncOp:
    """MLIR problem builder.

    Given a list of MLIR shaped types, build and return the MLIR FuncOp that
    implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, [types[-1]]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(
        func, [StringAttr.get(os.getenv('SANDBOX_INLINING', 'noinline'))],
        avx512=avx512)

    acc_type = types[-1].element_type
    with InsertionPoint(func.add_entry_block()):
      tensor_zero = func.arguments[2]
      if zero_at_each_iteration:
        zero = arith.ConstantOp(types[-1].element_type, 0.0)
        tensor_zero = linalg.FillOp(output=tensor_zero, value=zero)
      matmul = linalg.matmul(func.arguments[0],
                             func.arguments[1],
                             outs=[tensor_zero])
      # linalg.matmul returns a Value instead of OpView, so we have to manually
      # wrap it in a list here.
      std.ReturnOp([matmul])

    return func


# TODO: fold OpDSL definition and inferences into ProblemDefinition.
@linalg_structured_op
def add_bias_to_2d(I=TensorDef(T, S.M, S.N),
                   Bias=TensorDef(T, S.N),
                   O=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n)
  O[D.m, D.n] = I[D.m, D.n] + Bias[D.n]


class MatmulBiasAddProblem(ProblemDefinition):
  """ Problem definition for a fill + matmul + generic op."""

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the list of lists of shapes of the FuncOp operands. The
    FuncOp is responsible for distinguishing between input operands and results.
    """
    M, N, K = sizes["M"], sizes["N"], sizes["K"]
    return [
        [M, K],
        [K, N],
        [N],
        [M, N],
        [M, N],
    ]

  # TODO: tensors_np_builder and check_np.

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """MLIR types builder.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of elemental MLIR types, return MLIR tensor types of the
    shape expected by the function.
    """
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in \
            zip(shapes, list(types) + [types[-1]])]

  def build_problem_under_context_manager(
      self, name: str, types: Sequence[Type],
      zero_at_each_iteration: bool) -> builtin.FuncOp:
    """MLIR problem builder.

    Given a list of MLIR shaped types, build and return the MLIR FuncOp that
    implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, [types[-2]]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func,
                                  inplaceable=[False, False, False, True, True])
    attach_passthrough(
        func, [StringAttr.get(os.getenv('SANDBOX_INLINING', 'noinline'))],
        avx512=avx512)

    acc_type = types[-2].element_type
    with InsertionPoint(func.add_entry_block()):
      tensor_zero = func.arguments[3]
      if zero_at_each_iteration:
        zero = arith.ConstantOp(types[-1].element_type, 0.0)
        tensor_zero = linalg.FillOp(output=tensor_zero, value=zero)
      matmul = linalg.matmul(func.arguments[0],
                             func.arguments[1],
                             outs=[tensor_zero])
      bias_add = add_bias_to_2d(matmul,
                                func.arguments[2],
                                outs=[func.arguments[4]])
      # linalg.matmul returns a Value instead of OpView, so we have to manually
      # wrap it in a list here.
      std.ReturnOp([bias_add])

    return func
