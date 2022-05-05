import os
import numpy as np

from typing import Any, List, Mapping, Sequence, Tuple, Union

from iree.compiler.ir import *
from iree.compiler.dialects import arith, builtin, linalg, func

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = True


class CopyProblem(ProblemDefinition):
  """Benchmarking problem definition for copy.

  This requires a separate ProblemDefinition than Einsum on tensors
  to avoid the issues related to folding an identity generic op on
  tensors that just removes the computation we are interested in
  benchmarking.
  """

  def __init__(self, dims: List[str]):
    """Creates a new Copy with the given specification.
    """
    self.dims = dims

  @property
  def keys(self) -> List[str]:
    """Returns the list of parameter keys for the current problem definition."""
    return self.dims

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Constructs the tensor shapes given problem parameters."""
    operand_dims = [self.keys, self.keys]

    def shape_of_tensor(dims: str):
      return [sizes[k] for k in dims]

    return [shape_of_tensor(d) for d in operand_dims if d is not None]

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """Returns the GFLOp count given problem parameters."""
    return 0

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """Return the GByte count given problem parameters."""
    shapes = self.shapes_builder(sizes)
    gbyte_count = 0
    for type, shape in zip(types, shapes):
      gbyte_count += 1.e-9 * (np.prod(shape) * np.dtype(type).itemsize)
    return gbyte_count

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """Returns random NumPy suitable for calling the kernel."""
    shapes = [s if s else [1] for s in self.shapes_builder(sizes)]
    tensors = [
        realign(np.random.rand(*s).astype(t), byte_alignment=64)
        for s, t in zip(shapes, types)
    ]
    tensors[-1].fill(0.)
    return tensors

  def check_np(self, *args: np.dtype) -> None:
    """Checks whether the computation results mathch the reference impl.

    Given the list of NumPy arrays, computes the expected result and compares it
    with the actual result. Raises ValueError on mismatch.
    """
    input = args[0]
    output = args[-1]
    if not np.allclose(output, input):
      delta = output - input
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise ValueError(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """Returns the list of MLIR types for arguments of this computation."""
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in \
            zip(shapes, types)]

  def build_problem_under_context_manager(self,
                                          name: str,
                                          types: Sequence[Type],
                                          zero_at_each_iteration: bool = False):
    """Constructs MLIR that implements the einsum specification.

    Expects to operate under MLIR's context manager.

    Arguments:
    name: name of the MLIR function to generate (must be unique in its parent
      module).
    mlir_types: types of arguments of this computation.
    """
    global avx512

    bench = func.FuncOp(name, (types, [types[-1]]))
    inplaceable_attributes = [False] * len(types)
    inplaceable_attributes[-1] = True
    # TODO: need something much more flexible to add function argument attributes.
    attach_inplaceable_attributes(bench, inplaceable=inplaceable_attributes)
    attach_passthrough(
        bench, [StringAttr.get(os.getenv('SANDBOX_INLINING', 'noinline'))],
        avx512=avx512)

    with InsertionPoint(bench.add_entry_block()):
      input_tensor = bench.arguments[0]
      output_tensor = bench.arguments[-1]
      copy = linalg.copy(input_tensor, outs=[output_tensor])
      func.ReturnOp([copy])

    return bench
