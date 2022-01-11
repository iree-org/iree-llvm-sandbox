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
avx512 = True

DIMNAMES = 'MNKLPQRST'


class CopyNDProblem(ProblemDefinition):
  """ Problem definition for an n-dimensional copy problem."""

  def __init__(self, rank: int, op_builder: Callable):
    """Creates a problem definition for an n-dimensional copy.

    """
    self.__rank = rank
    self.__op_builder = op_builder

  @property
  def rank(self) -> int:
    return self.__rank

  @property
  def keys(self) -> List[str]:
    return list(DIMNAMES[:self.rank])

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the list of lists of shapes of the FuncOp operands. The
    FuncOp is responsible for distinguishing between input operands and results.
    """
    linear = [sizes[d] for d in self.keys]
    return [linear, linear]

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
    return float(
        np.prod(list(sizes.values())) *
        sum([np.dtype(ty).itemsize for ty in types])) / 1.e9

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
    #     for s, t in zip(shapes, types)
    # ]
    tensors[-1].fill(0.)
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
    return [RankedTensorType.get(s, t) for s, t in zip(shapes, mlir_types)]

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
      result = self.__op_builder(func.arguments[0], outs=[func.arguments[1]])
      std.ReturnOp([result])

    return func
