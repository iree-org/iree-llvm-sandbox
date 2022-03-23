from typing import Any, List, Mapping, Optional, Sequence, Union
from mlir.dialects.func import FuncOp

# Qualified import of only np.dtype for type checking.
np = __import__('numpy', fromlist=['dtype'])

# Qualified import of mlir.ir.Type to disambiguate with typing.Type.
ir = __import__('mlir.ir', fromlist=['Type'])


class ProblemDefinition:
  """ Generic problem definition interface."""

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the list of lists of shapes of the FuncOp operands. The
    FuncOp is responsible for distinguishing between input operands and results.
    """
    pass

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """GFlop builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the number of GFlops computed.
    """
    pass

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """GByte builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of data types, return the number of GBytes read or
    written.
    """
    pass

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """NumPy tensors building function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of NumPy elemental types, return constructed NP values of
    shapes given by `shape_builder` and specified elemental types.
    """
    pass

  def check_np(self, *tensors: np.dtype) -> None:
    """NumPy checking function.

    Given a list of NumPy values, check the precomputed results matches those of
    the expected reference implementation.
    """
    pass

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[ir.Type]) -> List[ir.Type]:
    """MLIR types builder.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of elemental MLIR types, return MLIR tensor types of the
    shape expected by the function.
    """
    pass

  def build_problem_under_context_manager(self, name: str,
                                          types: Sequence[ir.Type]) -> FuncOp:
    """MLIR problem builder.

    Given a list of MLIR shaped types, build and return the MLIR FuncOp that
    implements the desired computation on those types.
    """
    pass
