from typing import Any, List, Optional, Sequence, Type, Union

# Qualified import of only np.dtype for type checking.
np = __import__('numpy', fromlist=['dtype'])


class ProblemDefinition:
  """ Generic problem definition interface."""

  def shapes_builder(self, *args: int) -> List[List[int]]:
    """Shape builder function.

       Given a list of integer dimensions, return the list of lists of shapes
       of the FuncOp operands. The FuncOp is responsible for distinguishing
       between input operands and results.
    """
    pass

  def gflop_count_builder(self, *args: int) -> float:
    """GFlop builder function.

       Given a list of integer dimensions, return the number of GFlops computed.
    """
    pass

  def gbyte_count_builder(self, *args: int) -> float:
    """GByte builder function.

       Given a list of integer dimensions, return the number of GBytes read or
       written.
    """
    pass

  def tensors_np_builder(self, *args: Union[int, np.dtype]) -> List[np.dtype]:
    """NP tensors building function.

       Given a list of integer dimensions followed by per-operand NP elemental
       types, return constructed NP values of shapes given by `shaped_builder`
       and specified elemental types.
    """
    pass

  def check_np(self, *args: np.dtype) -> None:
    """NP checking function.

       Given a list of NP values, check the precomputed results matches those
       of the expected reference implementation.
    """
    pass

  def types_mlir_builder(self, *args: Union[int, Type]) -> List[Type]:
    """ MLIR types builder.

        Given a list of NP values, check the precomputed results matches those
        of the expected reference implementation.
    """
    pass

  def build_problem_under_context_manager(self, name: str, *args: Type):
    # TODO: -> FuncOp
    """MLIR problem builder.

       Given a flat list of MLIR types, build and return the MLIR FuncOp that
       implements the desired computation on those types.
    """
    pass
