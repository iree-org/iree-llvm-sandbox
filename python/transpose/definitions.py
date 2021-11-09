import sys, time

from collections.abc import Callable
from typing import Any, List, Optional, Sequence

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = False


class TransposeNDProblem(ProblemDefinition):
  """ Problem definition for a single n-dimensional transpose problem."""

  def __init__(self, permutation: List[int], op_builder: Callable):
    """Creates a problem definition for an n-dimensional transpose.

    The transposed dimensions are specified by the permutation array with the
    same format as `axes` in `numpy.transpose`. The rank (n) of the tranpose is
    the length of the permutation array. `op_builder` specifies the name of the
    function that constructs the corresponding Linalg op using OpDSL.
    """
    assert sorted(permutation) == list(range(len(permutation))), \
          'Expected permutation to be a permutation of [0,1,..,N-1]'
    self.__permutation = permutation
    self.__op_builder = op_builder

  def __transpose(self, items: List):
    """Returns the permutation of items using the current transpose."""
    return [items[i] for i in self.__permutation]

  def __partition_argument_list(self, items: List):
    """Splits items into parts containing integers and non-integers.

    The integers are expected to be the leading arguments in the item list.
    """
    for (i, arg) in enumerate(items):
      if not isinstance(arg, int):
        break
    return items[:i], items[i:]

  def shapes_builder(self, *sizes):
    """Shape builder function.

    Given a list of integer dimensions, return the list of lists of shapes of
    the FuncOp operands. The FuncOp is responsible for distinguishing between
    input operands and results.
    """
    return [[*sizes], self.__transpose([*sizes])]

  def gflop_count_builder(self, *sizes):
    """GFlop builder function.

    Given a list of integer dimensions, return the number of GFlops computed.
    """
    return float(0.)

  def gbyte_count_builder(self, *args: List):
    """GByte builder function.

    Given a list of integer dimensions followed by a list of data types, return
    the number of GBytes read or written.
    """
    sizes, types = self.__partition_argument_list(args)
    return float(np.prod(sizes)) * sum([np.dtype(ty).itemsize for ty in types
                                       ]) / 1.e9

  def tensors_np_builder(self, *args: List):
    sizes, np_types = self.__partition_argument_list(args)
    shapes = self.shapes_builder(*sizes)
    tensors = [
        np.random.rand(*shape).astype(ty)
        for shape, ty in zip(shapes, np_types)
    ]
    tensors[-1].fill(0.)
    return tensors

  def check_np(self, I, O):
    """NumPy checking function.

    Given a list of NumPy values, check the precomputed results matches those of
    the expected reference implementation.
    """
    if not np.allclose(O, np.transpose(I, axes=self.__permutation)):
      delta = O - np.transpose(I, axes=self.__permutation)
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise ValueError(f'Failed test with delta: {max_abs_delta}')

  def types_mlir_builder(self, *args: List):
    """MLIR types builder.

    Given a list of NP values, check the precomputed results matches those of
    the expected reference implementation.
    """
    sizes, types = self.__partition_argument_list(args)
    shapes = self.shapes_builder(*sizes)
    return [RankedTensorType.get(shape, ty) for shape, ty in zip(shapes, types)]

  def build_problem_under_context_manager(self, name, inp_mlir_type,
                                          out_mlir_type):
    """MLIR problem builder.

    Given a flat list of MLIR types, build and return the MLIR FuncOp that
    implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name,
                          ([inp_mlir_type, out_mlir_type], [out_mlir_type]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    with InsertionPoint(func.add_entry_block()):
      result = self.__op_builder(func.arguments[0], outs=[func.arguments[1]])
      std.ReturnOp([result])

    return func
