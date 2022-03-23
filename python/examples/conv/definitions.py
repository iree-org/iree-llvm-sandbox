import os, itertools

from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, func, linalg, scf, func, tensor

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

  @property
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

  def __infer_output_shape(self, sizes: Mapping[str, Any]) -> List[int]:
    """Compute the output shape given the list of problem parameters."""
    input_rank_dims_start, input_rank_dims_end = find_contiguous_rank_dims(
        self.__input_format)
    kernel_rank_dims_start, kernel_rank_dims_end = find_contiguous_rank_dims(
        self.__kernel_format)

    # 1. Non-DHW leading input dimensions except C.
    output_dims = [
        sizes[d]
        for d in self.__input_format[:input_rank_dims_start]
        if d != "C"
    ]

    # 2. Non-DHW leading kernel dimensions except C.
    for d in self.__kernel_format[:kernel_rank_dims_start]:
      if d == "C":
        continue
      output_dims.append(sizes[d])

    # 3. DHW dimensions.
    output_dims += [
        sizes[d]
        for d in self.__input_format[input_rank_dims_start:input_rank_dims_end]
    ]

    # 4. Non-DHW trailing input dimensions except C.
    output_dims += [
        sizes[d] for d in self.__input_format[input_rank_dims_end:] if d != "C"
    ]

    # 5. Non-DHW trailing kernel dimensions except C.
    for d in self.__kernel_format[kernel_rank_dims_end:]:
      if d == "C":
        continue
      output_dims.append(sizes[d])
    return output_dims

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Constructs the tensor shapes given problem parameters."""
    strides, dilations = sizes["strides"], sizes["dilations"]
    rank = len(self.__input_format) - 2

    # The input size is computed by increasing its rank-related dimensions to
    # accommodate kernel sizes with eventual strides and dilations, e.g.:
    #   dims(I) = [N,
    #             (H - 1) * SH + 1 + (KH - 1) * DH + 1,
    #             (W - 1) * SW + 1 + (KW - 1) * DW + 1,
    #.             C]
    #
    # Shape relations on rank-related dimensions:
    #      iw = [(ow - 1) * sw + 1] + [(kw - 1) * dw + 1] - 1
    #   => iw = [(ow - 1) * sw] + [(kw - 1) * dw] + 1
    #   => ow = [iw - (kw - 1) * dw - 1] / sw + 1
    input_shape = []
    for char in self.__input_format:
      if char in RANK_RELATED_DIMS[-rank:]:
        attribute_pos = RANK_RELATED_DIMS[-rank:].index(char)
        ow = sizes[char]
        sw = strides[attribute_pos]
        kw = sizes["K" + char]
        dw = dilations[attribute_pos]
        input_shape.append(((ow - 1) * sw + 1) + ((kw - 1) * dw + 1) - 1)
      else:
        input_shape.append(sizes[char])

    # The kernel size is derived directly from the corresponding parameters,
    # e.g.:
    #  dims(K) = [KH, KW, C, F].
    # C is provided for the input, take it there/
    kernel_shape = []
    for char in self.__kernel_format:
      if char not in RANK_RELATED_DIMS[-rank:]:
        kernel_shape.append(sizes[char])
      else:
        kernel_shape.append(sizes["K" + char])

    return [input_shape, kernel_shape, self.__infer_output_shape(sizes)]

  def stride_dilation_input_scaler(self):
    # Input scaler: we need to account for sizes and strides which involves
    # covering lattice computations. We just unroll a few common cases.
    scaler = []
    for stride, dilation in zip(self.__strides, self.__dilations):
      # If the stride is 1, we touch all input points along dim.
      if stride == 1:
        scaler.append(1)
      # If the stride is 2 and the dilation is a multiple of 2, we only touch
      # half of the input points along dim.
      # TODO: this should actually be a layout optimization in the compiler.
      elif stride == 2:
        if dilation % 2 is 0:
          # This also holds when kernel size is 1
          scaler.append(2)
        else:
          scaler.append(1)
      else:
        assert False, f'unsupport special case for stride = {stride}'
    return scaler

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """Returns the GFLOp count given problem parameters.

    Note: the problem parameters are always "output and kernel parameters", so
    we do not need special handling to account for strides and dilations:
    every output point is obtained by reducing across the kernel dimensions
    """
    return 2.0 * np.prod([
        sizes[k] for k in set(sizes.keys()) - set(["strides", "dilations"])
    ]) / 1.e9

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """Return the GByte count given problem parameters."""
    shapes = self.shapes_builder(sizes)

    input_shape, kernel_shape, output_shape = shapes
    input_np_type, kernel_np_type, output_np_type = types

    input_scaler = self.stride_dilation_input_scaler()
    input_gbytes = (1.e-9 * np.prod(input_shape) *
                    np.dtype(input_np_type).itemsize) / np.prod(input_scaler)
    kernel_gbytes = 1.e-9 * np.prod(kernel_shape) * np.dtype(
        kernel_np_type).itemsize
    # Output should only be counted as written because we are supposedly
    # overwriting the data without reading it.
    output_gbytes = 1.e-9 * np.prod(output_shape) * np.dtype(
        output_np_type).itemsize
    return input_gbytes + kernel_gbytes + output_gbytes

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """Returns random NumPy suitable for calling the kernel."""
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
    # Shape relations on rank-related dimensions:
    #      iw = [(ow - 1) * sw + 1] + [(kw - 1) * dw + 1] - 1
    #   => iw = [(ow - 1) * sw] + [(kw - 1) * dw] + 1
    #    => ow = [iw - (kw - 1) * dw - 1] / sw + 1
    iw = np.array(input_rank_dims)
    kw = np.array(kernel_rank_dims)
    dw = np.array(self.__dilations)
    sw = np.array(self.__strides)
    ones = np.ones(iw.shape, dtype=int)
    output_rank_dims = tuple((iw - (kw - ones) * dw - ones) // sw + 1)
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
          input_strides.append( \
              slice_input.strides[i] * self.__strides[ranked_pos])
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

      reference_O += np.tensordot(slice_input,
                                  slice_kernel,
                                  axes=([input_parallel_dim],
                                        [kernel_parallel_dim]))

    if not np.allclose(O, reference_O):
      delta = O - reference_O
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise ValueError(f"max_abs_delta: {max_abs_delta} -> FAILURE ")

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """Returns the list of MLIR types for arguments of this computation."""
    shapes = self.shapes_builder(sizes)
    return [RankedTensorType.get(s, t) for s, t in zip(shapes, types)]

  def build_problem_under_context_manager(
      self, name: str, mlir_types: Sequence[Type],
      zero_at_each_iteration: bool) -> func.FuncOp:
    """Constructs MLIR that implements the current convolution.

    Expects to operate under MLIR's context manager.

    Arguments:
    name: name of the MLIR function to generate (must be unique in its parent
      module).
    mlir_types: types of arguments of this computation.
    """
    global avx512

    output_type = mlir_types[-1]
    bench = func.FuncOp(name, (mlir_types, [output_type]))
    # TODO: need something much more flexible to add function argument attributes.
    attach_inplaceable_attributes(bench, inplaceable=[False, False, True])
    attach_passthrough(
        bench, [StringAttr.get(os.getenv('SANDBOX_INLINING', 'noinline'))],
        avx512=avx512)

    with InsertionPoint(bench.add_entry_block()):
      tensor_zero = bench.arguments[-1]
      if zero_at_each_iteration:
        zero = arith.ConstantOp(output_type.element_type, 0.0)
        tensor_zero = linalg.fill(zero, outs=[tensor_zero])
      conv = self.__op_builder(bench.arguments[0],
                               bench.arguments[1],
                               outs=[tensor_zero],
                               strides=self.__strides,
                               dilations=self.__dilations)
      func.ReturnOp([conv])

    return bench
