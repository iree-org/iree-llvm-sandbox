import sys, time

from collections.abc import Callable
from typing import Any, List, Optional, Sequence, Type

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std
from mlir.dialects.linalg.opdsl.lang import *

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = True

################################################################################
### Row Reduction 2D
################################################################################
#   Op def: (     m,     k )
#    Iters: ({Par(), Red()})
#             I       O
#   Layout: {{m, k}, {m}}


# TODO: fold OpDSL definition and inferences into ProblemDefinition.
@linalg_structured_op
def row_reduction_2d(
    A=TensorDef(T, S.M, S.N), B=TensorDef(T, S.M, output=True)):
  domain(D.m, D.n)
  B[D.m] += A[D.m, D.n]


class RowReduction2DProblem(ProblemDefinition):
  """ Problem definition for a single fill + row_reduction_2d problem."""

  def shapes_builder(self, M: int, K: int) -> List[List[int]]:
    """Shape builder function.

       Given a list of integer dimensions, return the list of lists of shapes
       of the FuncOp operands. The FuncOp is responsible for distinguishing
       between input operands and results.
    """
    return [[M, K], [M]]

  def gflop_count_builder(self, M: int, N: int) -> float:
    """GFlop builder function.

       Given a list of integer dimensions, return the number of GFlops computed.
    """
    return float(M * N) / float(1e9)

  def gbyte_count_builder(self, M: int, N: int, inp_np_type: np.dtype,
                          out_np_type: np.dtype) -> float:
    """GByte builder function.

       Given a list of integer dimensions, return the number of GBytes read or
       written.
    """
    return float(M * N * np.dtype(inp_np_type).itemsize +
                 M * np.dtype(out_np_type).itemsize) / float(1e9)

  def tensors_np_builder(self, M: int, K: int, input_np_type: np.dtype,
                         res_np_type: np.dtype) -> List[np.dtype]:
    shapes = self.shapes_builder(M, K)
    np_types = [input_np_type, res_np_type]
    tensors = [
        realign(np.random.rand(*s).astype(t), byte_alignment=64)
        for s, t in zip(shapes, np_types)
    ]
    # Uncomment to simplify debugging.
    # tensors = [
    #     realign(np.arange(1, np.prod(s) + 1).reshape(s).astype(t), \
    #             byte_alignment=64) \
    #     for s, t in zip(shapes, np_types)
    # ]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, A: np.dtype, B: np.dtype) -> None:
    """NP checking function.

       Given a list of NP values, check the precomputed results matches those
       of the expected reference implementation.
    """
    if not np.allclose(B, np.sum(A, axis=1)):
      delta = B - np.sum(A, axis=1)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, M: int, K: int, input_mlir_type: Type,
                         res_mlir_type: Type) -> List[Type]:
    """ MLIR types builder.

        Given a list of NP values, check the precomputed results matches those
        of the expected reference implementation.
    """
    compiled_function_element_types = [input_mlir_type, res_mlir_type]
    shapes = self.shapes_builder(M, K)
    return [RankedTensorType.get(s, t) for s, t in \
         zip(shapes, compiled_function_element_types)]

  def build_problem_under_context_manager(self, name: str,
                                          input_mlir_type: Type,
                                          res_mlir_type: Type):
    # TODO: -> FuncOp
    """MLIR problem builder.

       Given a flat list of MLIR types, build and return the MLIR FuncOp that
       implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name,
                          ([input_mlir_type, res_mlir_type], [res_mlir_type]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    output_elem_type = res_mlir_type.element_type
    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(output_elem_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[1], value=zero)
      result = row_reduction_2d(func.arguments[0], outs=[tensor_zero])
      # linalg.matmul returns a Value instead of OpView, so we have to manually
      # wrap it in a list here.
      std.ReturnOp([result])

    return func
