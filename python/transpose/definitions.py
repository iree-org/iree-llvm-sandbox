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
avx512 = False

################################################################################
### Transpose
################################################################################
#   Op def: (     m,     n )
#    Iters: ({Par(), Par()})
#             I       O
#   Layout: {{n, m}, {m, n}}


# TODO: fold OpDSL definition and inferences into ProblemDefinition.
@linalg_structured_op
def transpose_2d(
    I=TensorDef(T, S.N, S.M), O=TensorDef(T, S.M, S.N, output=True)):
  domain(D.m, D.n)
  O[D.m, D.n] = I[D.n, D.m]


class Transpose2dProblem(ProblemDefinition):
  """ Problem definition for a single transpose_2d problem."""

  def shapes_builder(self, M: int, N: int) -> List[List[int]]:
    """Shape builder function.

       Given a list of integer dimensions, return the list of lists of shapes
       of the FuncOp operands. The FuncOp is responsible for distinguishing
       between input operands and results.
    """
    return [[N, M], [M, N]]

  # TODO: Add a GB/s count builder and report that properly.
  # TODO: Take into account the data type, for now we assume 4B / element.
  def gflop_count_builder(self, M: int, N: int) -> float:
    """GFlop builder function.

       Given a list of integer dimensions, return the number of GFlops computed.
    """
    # 1 array read, 1 array written with data type 4B
    return float(2. * 4. * M * N) / float(1e9)

  def tensors_np_builder(self, M: int, N: int, input_np_type: np.dtype,
                         res_np_type: np.dtype) -> List[np.dtype]:
    shapes = self.shapes_builder(M, N)
    np_types = [input_np_type, res_np_type]
    tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, I: np.dtype, O: np.dtype) -> None:
    """NP checking function.

       Given a list of NP values, check the precomputed results matches those
       of the expected reference implementation.
    """
    if not np.allclose(O, np.transpose(I)):
      delta = O - np.transpose(I)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, M: int, N: int, input_mlir_type: Type,
                         res_mlir_type: Type) -> List[Type]:
    """ MLIR types builder.

        Given a list of NP values, check the precomputed results matches those
        of the expected reference implementation.
    """
    compiled_function_element_types = [input_mlir_type, res_mlir_type]
    shapes = self.shapes_builder(M, N)
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
      result = transpose_2d(func.arguments[0], outs=[func.arguments[1]])
      std.ReturnOp([result])

    return func
