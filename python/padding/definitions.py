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
class Padded_Conv1d_NWC_WCF_Problem(ProblemDefinition):
  """ Problem definition for fill + conv1d_nwc_wcf with padded input."""

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
    return (2.0 * N * W * C * KW * F) / 1.e9

  def gbyte_count_builder(self, N: int, W: int, C: int, KW: int, F: int,
                          stride: int, dilation: int, input_np_type: np.dtype,
                          kernel_np_type: np.dtype,
                          output_np_type: np.dtype) -> float:
    """GByte builder function.

       Given a list of integer dimensions, return the number of GBytes read or
       written.
    """
    shapes = self.shapes_builder(N, W, C, KW, F, stride, dilation)
    ro_gbytes = 1e-9 * (
        np.prod(shapes[0]) * np.dtype(input_np_type).itemsize +
        np.prod(shapes[1]) * np.dtype(kernel_np_type).itemsize)
    rw_gbytes = 2e-9 * (np.prod(shapes[2]) * np.dtype(output_np_type).itemsize)
    return ro_gbytes + rw_gbytes

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
