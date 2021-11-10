import itertools

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std, tensor

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

from . import ops

# TODO: Orthogonal configuration object.
avx512 = True

RANK_RELATED_DIMS = "DHW"


def find_contiguous_rank_dims(lst: str) -> Tuple[int, int]:
  """Returns a the positions of the rank-related dimensions in the list.

  Return a pair of values where the first value is the index of the first
  rank-related dimension and the second value is the index of the dimension
  immediately after the last rank-related dimension in the input format string.
  Expects rank-related dimensions to be contiguous in the format.

  Arguments:
  lst: convolution format string containing 'D', 'H' and 'W' to indicate
    rank-related dimensions.
  """
  start, end = None, None
  for i, char in enumerate(lst):
    if char in RANK_RELATED_DIMS:
      if start is None:
        start = i
      end = i
  return start, end + 1


class ConvolutionProblem(ProblemDefinition):
  """Benchmarking problem definition for 1/2/3D convolutions.

  Supports various convolution formats, to be specified at construction time,
  as long as the "image" dimensions (depth, height, width) are contiguous and
  provided in the same order. For example, NCHW is supported but NHCW or NCWH
  are not. Note that in practice, "image" dimension names are interchangeable.
  """

  def __init__(self, input_format: str, kernel_format: str,
               strides: Optional[List[int]], dilations: Optional[List[int]]):
    """Creates a new ConvolutionProblem definition with the given specification.

    The rank and kind of the convolution are specified by the formatting
    strings. Each character in the string corresponds to one dimension of the
    convolution, with the following characters supported:

      D: depth dimension of the image;
      H: height dimension of the image;
      W: width dimension of the image;
      N: number of the image (input only);
      C: input channels of the image;
      F: output channels of the image (kernel only).

    D and H characters may be missing to create a 2d or 1d convolution,
    respectively. The D, H, W characters must be provided in this order and be
    contiguous in the string. Other characters are mandatory and may appear
    anywhere in the format string.

    The format of the output is derived as:

      - all non-DHW dimensions of the input that precede DHW in the input format
        specification, in the same order, except C.
      - all non-DHW dimensions of the kernel hat precede DHW in the kernel
        format specification, in the same order, except C.
      - DHW dimensions;
      - all non-DHW dimensions of the input that follow DHW in the input format,
        in the same order, except C.
      - all non-DHW dimensions of the kernel that follow DHW in the kernel
        format, in the same order, except C.

    The convolution can have strides and dilations provided as extra arguments.

    The IR builder for the operation is expected to exist in the "ops" object
    (typically, module) visible in the scope under the name "conv_nd_ifmt_kfmt"
    where n is the confolution rank, ifmt is the lowercased input format and
    kfmt is the lowercased kernel format.

    Arguments:
    input_format: format of the input as described above.
    kernel_format: format of the kernel as described above.
    strides: convolution strides, if None, assumed to be all 1.
    dilations: convolution dilations, if None, assumed to be all 1.
    """
    rank = len(input_format) - 2
    assert len(kernel_format) == rank + 2

    if strides is None:
      strides = [1] * rank
    if dilations is None:
      dilations = [1] * rank

    assert len(strides) == rank
    assert len(dilations) == rank
    for char in input_format:
      assert char in "NCDHW", "Unexpected input format specifier."
    for char in kernel_format:
      assert char in "CFDHW", "Unexpected kernel format specifier."

    name = format(
        f"conv_{len(input_format) - 2}d_{input_format.lower()}_{kernel_format.lower()}"
    )
    assert name in ops.__dict__, "Operation not defined."

    self.__strides = strides
    self.__dilations = dilations
    self.__input_format = input_format
    self.__kernel_format = kernel_format
    self.__op_builder = ops.__dict__[name]

  def keys(self) -> List[str]:
    """Returns the list of parameter keys for the current problem definition."""
    result = list(self.__input_format)
    result += [
        "K" + char if char in "DHW" else char
        for char in self.__kernel_format
        if char.upper() != "C"
    ]
    result += ["strides", "dilations"]
    return result

  def __partition_argument_list(self, items: Sequence):
    """Splits items into parts containing integers and non-integers.

    The integers are expected to be the leading arguments in the item list.
    """
    for (i, arg) in enumerate(items):
      if not isinstance(arg, int):
        break
    return items[:i], items[i:]

  def __infer_output_shape(self, sizes: Sequence[int]) -> List[int]:
    """Compute the output shape given the list of problem parameters."""
    input_rank_dims_start, input_rank_dims_end = find_contiguous_rank_dims(
        self.__input_format)
    kernel_rank_dims_start, kernel_rank_dims_end = find_contiguous_rank_dims(
        self.__kernel_format)

    # 1. Non-DHW leading input dimensions except C.
    output_dims = [
        sizes[i]
        for i, d in enumerate(self.__input_format[:input_rank_dims_start])
        if d != "C"
    ]

    # 2. Non-DHW leading kernel dimensions except C.
    kernel_pos = 0
    for d in self.__kernel_format[:kernel_rank_dims_start]:
      if d == "C":
        continue
      output_dims.append(sizes[len(self.__input_format) + kernel_pos])
      kernel_pos += 1

    # 3. DHW dimensions.
    output_dims += [
        sizes[i] for i in range(input_rank_dims_start, input_rank_dims_end)
    ]

    # 4. Non-DHW trailing input dimensions except C.
    output_dims += [
        sizes[input_rank_dims_end + i]
        for i, d in enumerate(self.__input_format[input_rank_dims_end:])
        if d != "C"
    ]

    # 5. Non-DHW trailing kernel dimensions except C.
    kernel_pos = 0
    for d in self.__kernel_format[kernel_rank_dims_end:]:
      if d == "C":
        continue
      output_dims.append(sizes[len(self.__input_format) + kernel_rank_dims_end +
                               kernel_pos])
      kernel_pos += 1
    return output_dims

  def shapes_builder(self, *args: Union[int, List[int]]) -> List[List[int]]:
    """Constructs the tensor shapes given problem parameters."""
    sizes, attributes = self.__partition_argument_list(args)
    strides, dilations = attributes
    rank = len(self.__input_format) - 2

    # The input size is computed by increasing its rank-related dimensions to
    # accommodate kernel sizes with eventual strides and dilations, e.g.:
    #   dims(I) = [N, H * SH + KH * DH, W * SW + KW * DW, C].
    input_shape = []
    for pos, char in enumerate(self.__input_format):
      if char in RANK_RELATED_DIMS[-rank:]:
        attribute_pos = RANK_RELATED_DIMS[-rank:].index(char)
        kernel_pos = self.__kernel_format.replace("C", "").index(char)
        input_shape.append(sizes[pos] * strides[attribute_pos] +
                           sizes[rank + 2 + kernel_pos] *
                           dilations[attribute_pos])
      else:
        input_shape.append(sizes[pos])

    # The kernel size is derived directly from the corresponding parameters,
    # e.g.:
    #  dims(K) = [KH, KW, C, F].
    # C is provided for the input, take it there/
    kernel_shape = []
    kernel_pos = 0
    for char in self.__kernel_format:
      if char == "C":
        kernel_shape.append(sizes[self.__input_format.index("C")])
      else:
        kernel_shape.append(sizes[rank + 2 + kernel_pos])
        kernel_pos += 1

    return [input_shape, kernel_shape, self.__infer_output_shape(sizes)]

  def gflop_count_builder(self, *args: Union[int, List[int]]) -> float:
    """Returns the GFLOp count given problem parameters."""
    sizes, attributes = self.__partition_argument_list(args)
    return (2.0 * np.prod(sizes)) / 1.e9

  def gbyte_count_builder(self, *args: Union[int, List[int]]) -> float:
    """Supposed to return GByte count given problem parameters.

    Currently always returns null since we are not interested in memory behavior
    of convolutions yet.
    """
    return 0.0

  def tensors_np_builder(
      self, *args: Union[int, List[int], np.dtype]) -> List[np.dtype]:
    """Returns random NumPy suitable for calling the kernel."""
    sizes, other_args = self.__partition_argument_list(args)
    strides, dilations = other_args[0:2]
    np_types = other_args[2:]
    shapes = self.shapes_builder(*sizes, strides, dilations)
    tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
    tensors[-1].fill(0.)
    return tensors

  def check_np(self, I: np.dtype, K: np.dtype, O: np.dtype):
    """Checks whether the computation results correspond to the reference

    implementation.

    Given the list of NumPy arrays, computes the expected result and compares it
    with the actual result. Raises ValueError on mismatch.
    """
    reference_O = np.zeros(O.shape)

    input_rank_dims_start, input_rank_dims_end = find_contiguous_rank_dims(
        self.__input_format)
    kernel_rank_dims_start, kernel_rank_dims_end = find_contiguous_rank_dims(
        self.__kernel_format)

    # Compute the output rank-related dimensions by doing the computation
    # inverse to that of shape_builder. An alternative would be to extract them
    # from the shape of O but that requires knowing their position in the list.
    input_rank_dims = I.shape[input_rank_dims_start:input_rank_dims_end]
    kernel_rank_dims = K.shape[kernel_rank_dims_start:kernel_rank_dims_end]
    output_rank_dims = tuple(
        (np.array(input_rank_dims) -
         np.array(kernel_rank_dims) * np.array(self.__dilations)) //
        np.array(self.__strides))
    input_parallel_dim = self.__input_format[::-1].index("N")
    kernel_parallel_dim = self.__kernel_format[::-1].index("F")

    # Compute the convolution by taking (overlapping) output-sized slices of the
    # input that start at 0,1,...,KH, 0,1,...KW offsets, computing their
    # tensordot-product with the corresponding slices of the kernel, and
    # accumulating the result by addition. For example, for the NHWC, HWCF
    # convolution, the computation is
    #   O += I[:, kh:kh+H, kw:kw+W, :] . K[kh, kw, :, :] for kh in 0..KH,
    #                                                    for kw in 0..KW.
    # with H, W and C dimensions being reduced by tensordot and N, F remaining.
    for ks in itertools.product(*map(range, kernel_rank_dims)):
      input_slices = []
      ranked_pos = 0
      for char in self.__input_format:
        if char in RANK_RELATED_DIMS:
          # slice(a,b) is just a flexible form of a:b.
          input_slices.append(
              slice(
                  self.__dilations[ranked_pos] * ks[ranked_pos],
                  self.__dilations[ranked_pos] * ks[ranked_pos] +
                  output_rank_dims[ranked_pos]))
          ranked_pos += 1
        else:
          input_slices.append(slice(None))
      slice_input = I[tuple(input_slices)]

      input_strides = []
      ranked_pos = 0
      for i, char in enumerate(self.__input_format):
        if char in RANK_RELATED_DIMS:
          input_strides.append(slice_input.strides[i] *
                               self.__strides[ranked_pos])
          ranked_pos += 1
        else:
          input_strides.append(slice_input.strides[i])
      slice_input.strides = input_strides

      kernel_slices = []
      ranked_pos = 0
      for char in self.__kernel_format:
        if char in RANK_RELATED_DIMS:
          kernel_slices.append(ks[ranked_pos])
          ranked_pos += 1
        else:
          kernel_slices.append(slice(None))
      slice_kernel = K[tuple(kernel_slices)]

      reference_O += np.tensordot(
          slice_input,
          slice_kernel,
          axes=([input_parallel_dim], [kernel_parallel_dim]))

    if not np.allclose(O, reference_O):
      delta = O - reference_O
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise ValueError(f"max_abs_delta: {max_abs_delta} -> FAILURE ")

  def types_mlir_builder(self, *args: Union[int, List[int],
                                            Type]) -> List[Type]:
    """Returns the list of MLIR types for arguments of this computation."""
    sizes, other_args = self.__partition_argument_list(args)
    strides, dilations = other_args[0:2]
    element_types = other_args[2:]
    shapes = self.shapes_builder(*sizes, strides, dilations)
    return [RankedTensorType.get(s, t) for s, t in zip(shapes, element_types)]

  def build_problem_under_context_manager(self, name: str,
                                          *mlir_types: Type) -> builtin.FuncOp:
    """Constructs MLIR that implements the current convolution.

    Expects to operate under MLIR's context manager.

    Arguments:
    name: name of the MLIR function to generate (must be unique in its parent
      module).
    mlir_types: types of arguments of this computation.
    """
    global avx512

    output_type = mlir_types[-1]
    func = builtin.FuncOp(name, (mlir_types, [output_type]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(func, [StringAttr.get("noinline")], avx512=avx512)

    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(output_type.element_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[2], value=zero)
      conv = self.__op_builder(
          func.arguments[0],
          func.arguments[1],
          outs=[tensor_zero],
          strides=self.__strides,
          dilations=self.__dilations)
      std.ReturnOp([conv])

    return func
