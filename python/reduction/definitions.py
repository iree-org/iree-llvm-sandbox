import sys, time

from typing import Any, List, Mapping, Optional, Sequence

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

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

       Given a mapping between dimension names / op attributes and their numeric
       values, return the list of lists of shapes of the FuncOp operands. The
       FuncOp is responsible for distinguishing between input operands and
       results.
    """
    M, K = sizes["M"], sizes["K"]
    return [[M, K], [M]]

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """GFlop builder function.

       Given a mapping between dimension names / op attributes and their numeric
       values, return the number of GFlops computed.
    """
    M, K = sizes["M"], sizes["K"]
    return float(M * K) / float(1e9)

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """GByte builder function.

       Given a mapping between dimension names / op attributes and their numeric
       values, and a list of data types, return the number of GBytes read or
       written.
    """
    M, K = sizes["M"], sizes["K"]
    inp_np_type, out_np_type = types
    return float(M * K * np.dtype(inp_np_type).itemsize +
                 M * np.dtype(out_np_type).itemsize) / float(1e9)

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """NumPy tensors building function.

       Given a mapping between dimension names / op attributes and their numeric
       values, and a list of NumPy elemental types, return constructed NP values
       of shapes given by `shape_builder` and specified elemental types.
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
    #     for s, t in zip(shapes, types)
    # ]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, A: np.dtype, B: np.dtype) -> None:
    """NumPy checking function.

       Given a list of NumPy values, check the precomputed results matches those
       of the expected reference implementation.
    """
    if not np.allclose(B, np.sum(A, axis=1)):
      delta = B - np.sum(A, axis=1)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f"max_abs_delta: {max_abs_delta} -> FAILURE ")

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """ MLIR types builder.

        Given a mapping between dimension names / op attributes and their
        numeric values, and a list of elemental MLIR types, return MLIR tensor
        types of the shape expected by the function.
    """
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in zip(shapes, types)]

  def build_problem_under_context_manager(
      self, name: str, types: Sequence[Type]) -> builtin.FuncOp:
    """MLIR problem builder.

       Given a list of MLIR shaped types, build and return the MLIR FuncOp that
       implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, types[-1:]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    output_elem_type = types[-1].element_type
    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(output_elem_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[1], value=zero)
      result = row_reduction_2d(func.arguments[0], outs=[tensor_zero])
      # linalg.matmul returns a Value instead of OpView, so we have to manually
      # wrap it in a list here.
      std.ReturnOp([result])

    return func
