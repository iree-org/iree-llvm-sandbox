import os, sys, time

from typing import Any, List, Mapping, Optional, Sequence

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

  def __init__(self, WpadL: int, WpadR: int, stride: int, dilation: int):
    self.WpadL = WpadL
    self.WpadR = WpadR
    self.stride = stride
    self.dilation = dilation

  def ensure_stride_and_dilation(self, stride: int, dilation: int):
    assert dilation == self.dilation and stride == self.stride, \
        f"""unexpected dilation {dilation} (expected {self.dilation}) or
        stride {stride} (expected {self.stride}) mismatch.\n
        They are attributes and must be fixed at problem creation time
        """

  def shapes_builder(self, sizes: Mapping[str, Any]) -> List[List[int]]:
    """Shape builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the list of lists of shapes of the FuncOp operands. The
    FuncOp is responsible for distinguishing between input operands and results.
    """
    stride, dilation = sizes["stride"], sizes["dilation"]
    N, W, C, KW, F = sizes["N"], sizes["W"], sizes["C"], sizes["KW"], sizes["F"]
    self.ensure_stride_and_dilation(stride, dilation)
    return [[N, stride * W + dilation * KW - self.WpadL - self.WpadR, C], \
            [KW, C, F], \
            [N, W, F]]

  def gflop_count_builder(self, sizes: Mapping[str, Any]) -> float:
    """GFlop builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, return the number of GFlops computed.
    """
    stride, dilation = sizes["stride"], sizes["dilation"]
    self.ensure_stride_and_dilation(stride, dilation)
    return 2.0 * np.prod([
        sizes[k]
        for k in sizes.keys() - set(["strides", "dilations", "WpadL", "WpadR"])
    ]) / 1.e9

  def gbyte_count_builder(self, sizes: Mapping[str, Any],
                          types: Sequence[np.dtype]) -> float:
    """GByte builder function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of data types, return the number of GBytes read or
    written.
    """
    shapes = self.shapes_builder(sizes)
    input_np_type, kernel_np_type, output_np_type = types
    ro_gbytes = 1e-9 * (np.prod(shapes[0]) * np.dtype(input_np_type).itemsize +
                        np.prod(shapes[1]) * np.dtype(kernel_np_type).itemsize)
    rw_gbytes = 2e-9 * (np.prod(shapes[2]) * np.dtype(output_np_type).itemsize)
    return ro_gbytes + rw_gbytes

  def tensors_np_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[np.dtype]) -> List[np.dtype]:
    """NumPy tensors building function.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of NumPy elemental types, return constructed NP values of
    shapes given by `shape_builder` and specified elemental types.
    """
    stride, dilation = sizes["stride"], sizes["dilation"]
    self.ensure_stride_and_dilation(stride, dilation)
    shapes = self.shapes_builder(sizes)
    tensors = [np.random.rand(*s).astype(t) for s, t in zip(shapes, types)]
    tensors[len(tensors) - 1].fill(0.)
    return tensors

  def check_np(self, I: np.dtype, K: np.dtype, O: np.dtype) -> None:
    """NumPy checking function.

    Given a list of NumPy values, check the precomputed results matches those of
    the expected reference implementation.
    """
    # TODO: lift to __init__.
    N, W, F = np.shape(O)[0], np.shape(O)[1], np.shape(O)[2]
    KW, C = np.shape(K)[0], np.shape(K)[1]
    I2 = np.pad(I, ((0, 0), (self.WpadL, self.WpadR), (0, 0)), 'constant')

    O2 = np.copy(O)
    O2.fill(0.)

    for kw in range(KW):
      # N, in(W), C => N, in(W), C red
      # Only take a W slice, we'll scale the stride separately.
      slice_input = I2[ \
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

  def types_mlir_builder(self, sizes: Mapping[str, Any],
                         types: Sequence[Type]) -> List[Type]:
    """MLIR types builder.

    Given a mapping between dimension names / op attributes and their numeric
    values, and a list of elemental MLIR types, return MLIR tensor types of the
    shape expected by the function.
    """
    stride, dilation = sizes["stride"], sizes["dilation"]
    self.ensure_stride_and_dilation(stride, dilation)

    shapes = self.shapes_builder(sizes)
    func_types = [RankedTensorType.get(s, t) for s, t in zip(shapes, types)]

    N, W, KW, C = sizes["N"], sizes["W"], sizes["KW"], sizes["C"]
    padded_input_shape = [N, stride * W + dilation * KW, C]
    padded_input_type = RankedTensorType.get(padded_input_shape, types[0])

    return func_types + [padded_input_type]

  def build_problem_under_context_manager(
      self, name: str, types: Sequence[Type],
      zero_at_each_iteration: bool) -> builtin.FuncOp:
    """MLIR problem builder.

    Given a list of MLIR shaped types, build and return the MLIR FuncOp that
    implements the desired computation on those types.
    """
    global avx512

    # Actual benchmarked function called under entry_point_name.
    func = builtin.FuncOp(name, (types[:-1], [types[-2]]))
    # TODO: need something much more flexible to add func argument attributes.
    attach_inplaceable_attributes(func, inplaceable=[False, False, True])
    attach_passthrough(
        func, [StringAttr.get(os.getenv('SANDBOX_INLINING', 'noinline'))],
        avx512=avx512)

    output_element_type = types[-2].element_type

    index_type = IndexType.get()
    i64_type = IntegerType.get_signless(64)
    zero_attr = IntegerAttr.get(i64_type, 0)
    wpadl_attr = IntegerAttr.get(i64_type, self.WpadL)
    wpadr_attr = IntegerAttr.get(i64_type, self.WpadR)

    with InsertionPoint(func.add_entry_block()):
      tensor_zero = func.arguments[-1]
      if zero_at_each_iteration:
        zero = arith.ConstantOp(output_element_type, 0.0)
        tensor_zero = linalg.FillOp(output=tensor_zero, value=zero)

      padded_input = tensor.PadOp(
          result=types[-1],
          source=func.arguments[0],
          low=[],
          high=[],
          static_low=ArrayAttr.get([zero_attr, wpadl_attr, zero_attr]),
          static_high=ArrayAttr.get([zero_attr, wpadr_attr, zero_attr]),
          nofold=False)
      block = Block.create_at_start(padded_input.region, [index_type] * 3)
      with InsertionPoint(block):
        zero = arith.ConstantOp(output_element_type, 0.0)
        tensor.YieldOp(zero)

      conv = linalg.conv_1d_nwc_wcf(padded_input,
                                    func.arguments[1],
                                    outs=[tensor_zero],
                                    strides=[self.stride],
                                    dilations=[self.dilation])
      # linalg.Conv1DNwcWcfOp returns a Value instead of OpView, so we have to
      # manually wrap it in a list here.
      std.ReturnOp([conv])

    return func
