import sys, time

from typing import Any, Mapping, List, Optional, Sequence

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std
from mlir.dialects.linalg.opdsl.lang import *

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = False

################################################################################
### Copy
################################################################################
#   Op def: (     m,     n )
#    Iters: ({Par(), Par()})
#             I       O
#   Layout: {{m, n}, {m, n}}


# TODO: fold OpDSL definition and inferences into ProblemDefinition.
@linalg_structured_op
def copy_2d(I=TensorDef(T, S.M, S.N), O=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n)
  O[D.m, D.n] = I[D.m, D.n]


class Copy2DProblem(ProblemDefinition):
  """ Problem definition for a single copy_2d problem."""

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the list of lists of shapes of the FuncOp operands. The
    FuncOp is responsible for distinguishing between input operands and results.
    """
    M, N = sizes["M"], sizes["N"]
    return [[M, N], [M, N]]

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """GFlop builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the number of GFlops computed.
    """
    return float(0.)

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """GByte builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of data types, return the number of GBytes read or
    written.
    """
    M, N = sizes["M"], sizes["N"]
    inp_np_type, out_np_type = types
    return float(M * N * np.dtype(inp_np_type).itemsize +
                 M * N * np.dtype(out_np_type).itemsize) / float(1e9)

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """NumPy tensors building function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of NumPy elemental types, return constructed NP values of
    shapes given by `shape_builder` and specified elemental types.
    """
    M, N = sizes["M"], sizes["N"]
    shapes = self.shapes_builder(sizes)
    # For transpose it is better to use integer matrices and see what happens.
    tensors = [
        realign(np.arange(0, M * N).reshape(s).astype(t), byte_alignment=64)
        for s, t in zip(shapes, types)
    ]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, I: np.dtype, O: np.dtype) -> None:
    """NumPy checking function.

    Given a list of NumPy values, check the precomputed results matches those of
    the expected reference implementation.
    """
    if not np.allclose(O, I):
      delta = O - I
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         mlir_types: Sequence[Type]) -> List[Type]:
    """MLIR types builder.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of elemental MLIR types, return MLIR tensor types of the
    shape expected by the function.
    """
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in \
         zip(shapes, mlir_types)]

  def build_problem_under_context_manager(
      self, name: str, types: Sequence[Type]) -> builtin.FuncOp:
    """MLIR problem builder.

    Given a list of MLIR shaped types, build and return the MLIR FuncOp that
    implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, [types[-1]]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    output_elem_type = types[-1].element_type
    with InsertionPoint(func.add_entry_block()):
      result = copy_2d(func.arguments[0], outs=[func.arguments[1]])
      std.ReturnOp([result])

    return func
