import os
import numpy as np

from typing import Any, List, Mapping, Sequence, Tuple, Union

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, std

from .einsum import EinsumSpecification, make_einsum
from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = True


class EinsumProblem(ProblemDefinition):
  """Benchmarking problem definition for einsum.

  Supports one-operand and two-operand einsum specifications. The textual
  specification of the operation is similar to that of np.einsum.
  """

  def __init__(self, specification: str, domain: str, flop_count_per_iter: int):
    """Creates a new EinsumProblem with the given specification.

    The specification is a string of the format:
    `<lhs> ',' (<rhs>)? ('->' <out>)?`
    where <lhs>, <rhs> and <out> are contiguous lists of lowercase letters
    indicating dimensions. One-operand specifications define only a lhs operand
    and skip the rhs operand. Dimensions that appear in both lhs and rhs are
    reduction dimensions and must not appear in the output. The remaining
    dimensions must appear in the output. Two-operand specifications may omit
    the output. In this case, the output dimensions are inferred by taking all
    non-reduction lhs and rhs dimensions and sorting them alphabetically.

    Arguments:
    specification: textual specification of the einsum.
    domain: textual specification of the einsum dimension in iteration order.
    flop_count_per_iter: floating-point operations executed per iteration.
    """
    self.specification = EinsumSpecification(specification, domain)
    self.flop_count_per_iter = flop_count_per_iter

  @property
  def keys(self) -> List[str]:
    """Returns the list of parameter keys for the current problem definition."""
    return list(self.specification.output_dims +
                self.specification.reduction_dims)

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Constructs the tensor shapes given problem parameters."""
    operand_dims = [
        self.specification.lhs_dims, self.specification.rhs_dims,
        self.specification.output_dims
    ]

    def shape_of_tensor(dims: str):
      return [sizes[k] for k in dims]

    return [shape_of_tensor(d) for d in operand_dims if d is not None]

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """Returns the GFLOp count given problem parameters."""
    return self.flop_count_per_iter * np.prod(list(sizes.values())) / 1.e9

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """Return the GByte count given problem parameters."""
    shapes = self.shapes_builder(sizes)
    gbyte_count = 0
    for type, shape in zip(types, shapes):
      gbyte_count += 1.e-9 * (np.prod(shape) * np.dtype(type).itemsize)
    # Reduction operators read and write the output.
    if self.specification.reduction_dims:
      gbyte_count += 1.e-9 * \
          (np.prod(shapes[-1]) * np.dtype(types[-1]).itemsize)
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
    """Checks whether the computation results correspond to the reference

    implementation.

    Given the list of NumPy arrays, computes the expected result and compares it
    with the actual result. Raises ValueError on mismatch.
    """
    output = args[-1]
    reference_output = np.einsum(str(self.specification), *args[:-1])
    if not np.allclose(output, reference_output):
      delta = output - reference_output
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise ValueError(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """Returns the list of MLIR types for arguments of this computation."""
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in \
            zip(shapes, types)]

  def build_problem_under_context_manager(self, name: str,
                                          types: Sequence[Type],
                                          zero_at_each_iteration: bool):
    """Constructs MLIR that implements the einsum specification.

    Expects to operate under MLIR's context manager.

    Arguments:
    name: name of the MLIR function to generate (must be unique in its parent
      module).
    mlir_types: types of arguments of this computation.
    """
    global avx512

    func = builtin.FuncOp(name, (types, [types[-1]]))
    inplaceable_attributes = [False] * len(types)
    inplaceable_attributes[-1] = True
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=inplaceable_attributes)
    attach_passthrough(
        func, [StringAttr.get(os.getenv('SANDBOX_INLINING', 'noinline'))],
        avx512=avx512)

    with InsertionPoint(func.add_entry_block()):
      output_tensor = func.arguments[-1]
      if self.specification.reduction_dims and zero_at_each_iteration:
        zero = arith.ConstantOp(types[-1].element_type, -0.0)
        output_tensor = linalg.FillOp(output=func.arguments[-1], value=zero)
      print('Einsum spec: ', str(self.specification))
      einsum_op = make_einsum(self.specification)(*func.arguments[:-1],
                                                  outs=[output_tensor])
      std.ReturnOp([einsum_op])

    return func
