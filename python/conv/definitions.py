import sys, time

from collections.abc import Callable
from typing import Any, List, Optional, Sequence, Type

import numpy as np

from mlir.ir import *
from mlir.dialects import arith, builtin, linalg, scf, std, tensor

from ..core.compilation import attach_inplaceable_attributes, attach_passthrough
from ..core.problem_definition import *
from ..core.utils import *

# TODO: Orthogonal configuration object.
avx512 = True

################################################################################
### Conv1d_NWC_WCF
################################################################################
#   Op def: (     n,     w,     c,    kw,    f  )
#    Iters: ({Par(), Par(), Par(), Red(), Red()})
#                          I                            K          O
#   Layout: {{n, strideW * w + dilationW * kw, c}, {kw, c, f}, {n, w, f}}
class Conv1d_NWC_WCF_Problem(ProblemDefinition):
  """ Problem definition for a single fill + conv1d_nwc_wcf problem."""

  stride: int
  dilation: int

  def __init__(self, stride: int, dilation: int):
    self.stride = stride
    self.dilation = dilation

  def ensure_stride_and_dilation(self, stride: int, dilation: int):
    assert dilation == self.dilation and stride == self.stride, \
        f"""unexpected dilation {dilation} (expected {self.dilation}) or
        stride {stride} (expected {self.stride}) mismatch.\n
        They are attributes and must be fixed at problem creation time
        """

  def shapes_builder(self, N: int, W: int, C: int, KW: int, F: int, stride: int,
                     dilation: int) -> List[List[int]]:
    """Shape builder function.

       Given a list of integer dimensions, return the list of lists of shapes
       of the FuncOp operands. The FuncOp is responsible for distinguishing
       between input operands and results.
    """
    self.ensure_stride_and_dilation(stride, dilation)
    return [[N, stride * W + dilation * KW, C], \
            [KW, C, F], \
            [N, W, F]]

  def gflop_count_builder(self, N: int, W: int, C: int, KW: int, F: int,
                          stride: int, dilation: int) -> float:
    """GFlop builder function.

       Given a list of integer dimensions, return the number of GFlops computed.
    """
    self.ensure_stride_and_dilation(stride, dilation)
    return (2.0 * N * W * C * KW * F) / 1e9

  def tensors_np_builder(self, N: int, W: int, C: int, KW: int, F: int,
                         stride: int, dilation: int, input_np_type: np.dtype,
                         kernel_np_type: np.dtype,
                         output_np_type: np.dtype) -> List[np.dtype]:
    """NP tensors building function.

       Given a list of integer dimensions followed by per-operand NP elemental
       types, return constructed NP values of shapes given by `shaped_builder`
       and specified elemental types.
    """
    self.ensure_stride_and_dilation(stride, dilation)
    shapes = self.shapes_builder(N, W, C, KW, F, stride, dilation)
    np_types = [input_np_type, kernel_np_type, output_np_type]
    tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, I: np.dtype, K: np.dtype, O: np.dtype) -> None:
    """NP checking function.

       Given a list of NP values, check the precomputed results matches those
       of the expected reference implementation.
    """
    # TODO: lift to __init__.
    N, W, F = np.shape(O)[0], np.shape(O)[1], np.shape(O)[2]
    KW, C = np.shape(K)[0], np.shape(K)[1]
    O2 = np.copy(O)
    O2.fill(0.)

    for kw in range(KW):
      # N, in(W), C => N, in(W), C red
      # Only take a W slice, we'll scale the stride separately.
      slice_input = I[ \
          :, \
          self.dilation * kw : self.dilation * kw + W, \
          :  \
      ]
      slice_input.strides = [ \
        slice_input.strides[0], \
        slice_input.strides[1] * self.stride, \
        slice_input.strides[2] \
      ]
      # KW, C, F -> C red, F
      slice_filter = K[kw, :, :]
      O2 = O2 + np.tensordot(slice_input, slice_filter, axes=([2], [0]))

    if not np.allclose(O, O2):
      delta = O - O2
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, N: int, W: int, C: int, KW: int, F: int,
                         stride: int, dilation: int, input_mlir_type: Type,
                         kernel_mlir_type: Type,
                         output_mlir_type: Type) -> List[Type]:
    """ MLIR types builder.

        Given a list of NP values, check the precomputed results matches those
        of the expected reference implementation.
    """
    self.ensure_stride_and_dilation(stride, dilation)
    compiled_function_element_types = [
        input_mlir_type, kernel_mlir_type, output_mlir_type
    ]
    shapes = self.shapes_builder(N, W, C, KW, F, stride, dilation)
    return [RankedTensorType.get(s, t) for s, t in \
         zip(shapes, compiled_function_element_types)]

  def build_problem_under_context_manager(self, name: str,
                                          input_mlir_type: Type,
                                          kernel_mlir_type: Type,
                                          output_mlir_type: Type):
    # TODO: -> FuncOp
    """MLIR problem builder.

       Given a flat list of MLIR types, build and return the MLIR FuncOp that
       implements the desired computation on those types.
    """
    global avx512

    types = [input_mlir_type, kernel_mlir_type, output_mlir_type]

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, [output_mlir_type]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    output_type = output_mlir_type.element_type
    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(output_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[2], value=zero)
      conv = linalg.conv_1d_nwc_wcf(
          func.arguments[0],
          func.arguments[1],
          outs=[tensor_zero],
          strides=[self.stride],
          dilations=[self.dilation])
      # linalg.Conv1DNwcWcfOp returns a Value instead of OpView, so we have to
      # manually wrap it in a list here.
      std.ReturnOp([conv])

    return func


################################################################################
### Conv2d_NHWC_HWCF
################################################################################
#   Op def: (     n,     h,     w,     c,    kh,    kw,    f  )
#    Iters: ({Par(), Par(), Par(), Par(), Red(), Red(), Red()})
#                          I                            K          O
#   Layout: {{n, strideH * h + dilationH * kh, strideW * w + dilationW * kw, c},
#            {kh, kw, c, f},
#            {n, h, w, f}}
class Conv2d_NHWC_HWCF_Problem(ProblemDefinition):
  """ Problem definition for a single fill + conv2d_nhwc_hwcf problem."""

  strides: List[int]
  dilations: List[int]

  def __init__(self, dilations: List[int], strides: List[int]):
    self.strides = strides
    self.dilations = dilations
    assert len(self.strides) == 2, 'strides must be a 2-element List[int]'
    assert len(self.dilations) == 2, 'dilations must be a 2-element List[int]'

  def ensure_strides_and_dilations(self, strides: List[int],
                                   dilations: List[int]):
    assert dilations == self.dilations and strides == self.strides, \
        f"""unexpected dilations {dilations} (expected {self.dilations}) or
        strides {strides} (expected {self.strides}) mismatch.\n
        They are attributes and must be fixed at problem creation time
        """

  def shapes_builder(self, N: int, H: int, W: int, C: int, KH: int, KW: int,
                     F: int, strides: List[int],
                     dilations: List[int]) -> List[List[int]]:
    """Shape builder function.

       Given a list of integer dimensions, return the list of lists of shapes
       of the FuncOp operands. The FuncOp is responsible for distinguishing
       between input operands and results.
    """
    self.ensure_strides_and_dilations(strides, dilations)
    return [[N, strides[0] * H + dilations[0] * KH, strides[1] * W + dilations[1] * KW, C], \
            [KH, KW, C, F], \
            [N, H, W, F]]

  def gflop_count_builder(self, N: int, H: int, W: int, C: int, KH: int,
                          KW: int, F: int, strides: List[int],
                          dilations: List[int]) -> float:
    """GFlop builder function.

       Given a list of integer dimensions, return the number of GFlops computed.
    """
    self.ensure_strides_and_dilations(strides, dilations)
    return (2.0 * N * H * W * C * KH * KW * F) / 1e9

  def tensors_np_builder(self, N: int, H: int, W: int, C: int, KH: int, KW: int,
                         F: int, strides: List[int], dilations: List[int],
                         input_np_type: np.dtype, kernel_np_type: np.dtype,
                         output_np_type: np.dtype) -> List[np.dtype]:
    """NP tensors building function.

       Given a list of integer dimensions followed by per-operand NP elemental
       types, return constructed NP values of shapes given by `shaped_builder`
       and specified elemental types.
    """
    self.ensure_strides_and_dilations(strides, dilations)
    shapes = self.shapes_builder(N, H, W, C, KH, KW, F, strides, dilations)
    np_types = [input_np_type, kernel_np_type, output_np_type]
    tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, np_types)]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, I: np.dtype, K: np.dtype, O: np.dtype) -> None:
    """NP checking function.

       Given a list of NP values, check the precomputed results matches those
       of the expected reference implementation.
    """
    # TODO: lift to __init__.
    N, H, W, F = \
        np.shape(O)[0], np.shape(O)[1], np.shape(O)[2], np.shape(O)[3]
    KH, KW, C = np.shape(K)[0], np.shape(K)[1], np.shape(K)[2]
    O2 = np.copy(O)
    O2.fill(0.)

    import itertools
    for (kh, kw) in itertools.product(range(KH), range(KW)):
      # N, in(H), in(W), C => N, in(H), in(W), C red
      # Only take a H x W slice, we'll scale the stride separately.
      slice_input = I[ \
        :, \
        self.dilations[0] * kh : self.dilations[0] * kh + H, \
        self.dilations[1] * kw : self.dilations[1] * kw + W, \
        : \
      ]
      slice_input.strides = [ \
        slice_input.strides[0], \
        slice_input.strides[1] * self.strides[0], \
        slice_input.strides[2] * self.strides[1], \
        slice_input.strides[3] \
      ]
      # KH, KW, C, F -> C red, F
      slice_filter = K[kh, kw, :, :]
      O2 = O2 + np.tensordot(slice_input, slice_filter, axes=([3], [0]))

    if not np.allclose(O, O2):
      delta = O - O2
      max_abs_delta = max(delta.max(), delta.min(), key=abs)
      raise Exception(f'max_abs_delta: {max_abs_delta} -> FAILURE ')

  def types_mlir_builder(self, N: int, H: int, W: int, C: int, KH: int, KW: int,
                         F: int, strides: List[int], dilations: List[int],
                         input_mlir_type: Type, kernel_mlir_type: Type,
                         output_mlir_type: Type) -> List[Type]:
    """ MLIR types builder.

        Given a list of NP values, check the precomputed results matches those
        of the expected reference implementation.
    """
    self.ensure_strides_and_dilations(strides, dilations)
    compiled_function_element_types = [
        input_mlir_type, kernel_mlir_type, output_mlir_type
    ]
    shapes = self.shapes_builder(N, H, W, C, KH, KW, F, strides, dilations)
    return [RankedTensorType.get(s, t) for s, t in \
         zip(shapes, compiled_function_element_types)]

  def build_problem_under_context_manager(self, name: str,
                                          input_mlir_type: Type,
                                          kernel_mlir_type: Type,
                                          output_mlir_type: Type):
    # TODO: -> FuncOp
    """MLIR problem builder.

       Given a flat list of MLIR types, build and return the MLIR FuncOp that
       implements the desired computation on those types.
    """
    global avx512

    types = [input_mlir_type, kernel_mlir_type, output_mlir_type]

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types, [output_mlir_type]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(func, [StringAttr.get('noinline')], avx512=avx512)

    num_results = len(func.type.results)
    output_type = output_mlir_type.element_type
    with InsertionPoint(func.add_entry_block()):
      zero = arith.ConstantOp(output_type, 0.0)
      tensor_zero = linalg.FillOp(output=func.arguments[2], value=zero)
      conv = linalg.conv_2d_nhwc_hwcf(
          func.arguments[0],
          func.arguments[1],
          outs=[tensor_zero],
          strides=self.strides,
          dilations=self.dilations)
      std.ReturnOp([conv])

    return func
