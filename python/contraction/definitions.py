import numpy as np

from typing import List, Sequence, Tuple, Union

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

  Currently only supports two-operand contractions. The textual specification of
  the contraction to perform is similar to that of np.einsum.
  """

  def __init__(self, specification: str):
    """Creates a new EinsumProblem with the given specification.

    The specification is a string of the format `<lhs> ',' <rhs> ('->' <out>)?`
    where <lhs>, <rhs> and <out> are contiguous lists of lowercase letters
    indicating dimensions. Dimensions that appear in both lhs and rhs are
    reduction dimensions and must not appear in the output. The remaining
    dimensions must appear in the output. If the output is omitted, it is
    inferred by taking all non-reduction lhs and rhs dimensions and sorting them
    alphabetically.

    Arguments:
    specification: textual specification of the einsum.
    """
    self.specification = EinsumSpecification(specification)

  def keys(self) -> List[str]:
    """Returns the list of parameter keys for the current problem definition."""
    return list(self.specification.output_dims +
                self.specification.reduction_dims)

  def __partition_argument_list(self,
                                items: Sequence) -> Tuple[Sequence, Sequence]:
    """Splits items into parts containing integers and non-integers.

    The integers are expected to be the leading arguments in the item list.
    """
    for (i, arg) in enumerate(items):
      if not isinstance(arg, int):
        break
    else:
      return items, ()
    return items[:i], items[i:]

  def shapes_builder(self, *args: Union[int]) -> List[List[int]]:
    """Constructs the tensor shapes given problem parameters."""
    all_keys = self.keys()

    def shape_of_tensor(name: str):
      return [
          args[all_keys.index(d)] for d in getattr(self.specification, name)
      ]

    return [
        shape_of_tensor(name)
        for name in ['lhs_dims', 'rhs_dims', 'output_dims']
    ]

  def gflop_count_builder(self, *args: Union[int]) -> float:
    """Returns the GFLOp count given problem parameters."""
    sizes, types = self.__partition_argument_list(args)
    return 2.0 * np.prod(sizes) / 1.e9

  def gbyte_count_builder(self, *args: Union[int, np.dtype]) -> float:
    """Return the GByte count given problem parameters."""
    sizes, types = self.__partition_argument_list(args)
    lhs_type, rhs_type, output_type = types
    lhs_shape, rhs_shape, output_shape = self.shapes_builder(*sizes)
    ro_gbytes = 1.e-9 * (
        np.prod(lhs_shape) * np.dtype(lhs_type).itemsize +
        np.prod(rhs_shape) * np.dtype(rhs_type).itemsize)
    rw_gbytes = 2.e-9 * np.prod(output_shape) * np.dtype(output_type).itemsize
    return ro_gbytes + rw_gbytes

  def tensors_np_builder(self, *args: Union[int, np.dtype]) -> List[np.dtype]:
    """Returns random NumPy suitable for calling the kernel."""
    sizes, types = self.__partition_argument_list(args)
    shapes = self.shapes_builder(*sizes)
    tensors = [
        np.random.rand(*s).astype(t) for s, t in zip(shapes[:-1], types[:-1])
    ]
    return tensors + [np.zeros(shapes[-1]).astype(types[-1])]

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

  def types_mlir_builder(self, *args: Union[int, Type]) -> List[Type]:
    """Returns the list of MLIR types for arguments of this computation."""
    sizes, types = self.__partition_argument_list(args)
    shapes = self.shapes_builder(*sizes)
    return [RankedTensorType.get(s, t) for s, t in \
            zip(shapes, types)]

  def build_problem_under_context_manager(self, name: str, *types: Type):
    """Constructs MLIR that implements the current convolution.

    Expects to operate under MLIR's context manager.

    Arguments:
    name: name of the MLIR function to generate (must be unique in its parent
      module).
    mlir_types: types of arguments of this computation.
    """
    global avx512

    func = builtin.FuncOp(name, (types, [types[-1]]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(types[-1].element_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[-1], value=zero)
      contraction = make_einsum(str(self.specification))(
          *func.arguments[:-1], outs=[tensor_zero])
      std.ReturnOp([contraction])

    return func
