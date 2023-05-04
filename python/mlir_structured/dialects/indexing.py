# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import operator
import re
from copy import deepcopy
from functools import cached_property, lru_cache, partialmethod
from typing import Union, Tuple, Optional, NamedTuple, Sequence, Any, List

import numpy as np

from . import arith, tensor
from ._arith_ops_ext import _is_integer_like_type
from ._indexing_ops_gen import *
from ._ods_common import get_op_result_or_value
from ._structured_transform_ops_ext import _get_int_int_array_attr
from .linalg.opdsl.lang.emitter import _is_floating_point_type
from .._mlir_libs._structuredDialects.indexing import *
from ..ir import (
    Value,
    Type,
    RankedTensorType,
    ShapedType,
    Operation,
    OpView,
    DenseElementsAttr,
    register_attribute_builder,
    IntegerAttr,
    IntegerType,
    F16Type,
    F32Type,
    F64Type,
    FloatAttr,
    IndexType,
)

# from .._mlir_libs import _mlirStructuredPasses as _cextStructuredPasses

res_val_reg = re.compile(r"(%\w+) =")

# The classes in this file implement syntactic sugar for creating MLIR corresponding to arithmetic operations
# on `tensor`s (MLIR Value whose type is `tensor<...>`) and `scalar`s (`f*` floats, `i*` integers, and `index`).
#
# The implementation uses the Python operator overloading mechanism to overload various "dunder" methods (`__add__`, `__mul__`, etc.)
# to dispatch to the corresponding MLIR builders (`arith.AddFOp`, `arith.MulFOp`, etc.) through their Python bindings.
# The classes also provide some convenience factory methods and properties for accessing metadata (data type, shape, etc.).


def infer_mlir_type(py_val: Union[int, float, bool, np.ndarray]) -> Type:
  if isinstance(py_val, bool):
    return IntegerType.get_signless(1)
  elif isinstance(py_val, int):
    return IntegerType.get_signless(64)
  elif isinstance(py_val, float):
    return F64Type.get()
  elif isinstance(py_val, np.ndarray):
    return {
        np.int8: IntegerType.get_signless(8),
        np.int16: IntegerType.get_signless(16),
        np.int32: IntegerType.get_signless(32),
        np.int64: IntegerType.get_signless(64),
        np.uintp: IndexType.get(),
        np.float16: F16Type.get(),
        np.float32: F32Type.get(),
        np.float64: F64Type.get(),
    }[py_val.dtype.type]
  else:
    raise NotImplementedError(
        f"Unsupported Python value {py_val=} with type {type(py_val)}")


def constant(
    cst: Union[int, float, bool, np.ndarray],
    type: Optional[Type] = None,
    index: Optional[bool] = None,
) -> arith.ConstantOp:
  assert isinstance(cst, (int, float, bool, np.ndarray))
  if index is not None and index:
    type = IndexType.get()
  elif type is None:
    type = infer_mlir_type(cst)
  assert type is not None

  if isinstance(cst, np.ndarray):
    type = RankedTensorType.get(cst.shape, type)
    cst = DenseElementsAttr.get(
        cst,
        type=type,
    )

  cst = arith.ConstantOp(type, cst)

  return cst


# TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases
class ArithValueMeta(type(Value)):

  def __call__(cls, *args, **kwargs):
    if len(args) != 1:
      raise ValueError("Only one non-kw arg supported.")
    arg = args[0]
    if isinstance(arg, (OpView, Operation, Value)):
      if isinstance(arg, (Operation, OpView)):
        assert len(arg.results) == 1
      val = get_op_result_or_value(arg)
      arg_copy = None
    elif isinstance(arg, (int, float, bool, np.ndarray)):
      dtype = kwargs.get("dtype")
      index = kwargs.get("index")
      arg_copy = deepcopy(arg)
      val = constant(arg, dtype, index).result
    else:
      raise NotImplementedError(
          f"{cls.__name__} doesn't support wrapping {arg}.")

    cls_obj = cls.__new__(cls, val)
    cls.__init__(cls_obj, val, arg_copy)
    return cls_obj


class ArithValue(metaclass=ArithValueMeta):

  def __new__(cls, val):
    # The call chain after this is the __init__ here *and then* Value.__init__
    return super().__new__(cls, val)

  def __init__(self,
               val,
               arg: Optional[Union[int, float, bool, np.ndarray]] = None):
    self.__arg = arg
    super().__init__(val)

  @lru_cache(maxsize=1)
  def is_constant(self) -> bool:
    return isinstance(self.owner.opview, arith.ConstantOp)

  @lru_cache(maxsize=1)
  def __str__(self):
    s = res_val_reg.findall(super().__str__())
    assert len(s) == 1
    if self.is_constant():
      v = str(self.literal_value)
      v = f"{v[:10]}..." if len(v) > 10 else v
      return f"{self.__class__.__name__}({s[0]}, {self.type}, {v})"
    else:
      return f"{self.__class__.__name__}({s[0]}, {self.type})"

  @lru_cache(maxsize=1)
  def __repr__(self):
    return str(self)

  def __binary_op(self, other: "Tensor", op: str) -> "Tensor":
    assert op in {"add", "sub", "mul"}
    if self.type != other.type:
      raise ValueError(f"{self=} {other=} must have the same type.")

    if self.is_constant() and other.is_constant():
      lhs, rhs = self.literal_value, other.literal_value
      op = operator.attrgetter(op)(operator)
    else:
      op = op.capitalize()
      lhs, rhs = self, other
      if _is_floating_point_type(self.dtype):
        op = getattr(arith, f"{op}FOp")
      elif _is_integer_like_type(self.dtype):
        op = getattr(arith, f"{op}IOp")
      else:
        raise NotImplementedError(f"Unsupported '{op}' operands: {lhs}, {rhs}")
    return self.__class__(op(lhs, rhs))

  __add__ = partialmethod(__binary_op, op="add")
  __sub__ = partialmethod(__binary_op, op="sub")
  __mul__ = partialmethod(__binary_op, op="mul")


class Scalar(ArithValue, ScalarValue):

  @cached_property
  def dtype(self) -> Type:
    return self.type

  @cached_property
  def literal_value(self) -> Union[int, float, bool]:
    if not self.is_constant():
      raise ValueError("Can't build literal from non-constant Scalar")
    return self.owner.opview.literal_value


class Tensor(ArithValue, TensorValue):

  @cached_property
  def literal_value(self) -> np.ndarray:
    if not self.is_constant():
      raise ValueError("Can't build literal from non-constant Tensor")
    return np.array(DenseElementsAttr(self.owner.opview.value), copy=False)

  @cached_property
  def _shaped_type(self) -> ShapedType:
    return ShapedType(self.type)

  @cached_property
  def shape(self) -> Tuple[int, ...]:
    if not self._shaped_type.has_static_shape:
      raise ValueError("Only static shapes currently supported.")
    return tuple(self._shaped_type.shape)

  @cached_property
  def dtype(self) -> Type:
    return self._shaped_type.element_type

  @classmethod
  def empty(
      cls,
      shape: Union[list[Union[int, Value]], tuple[Union[int, Value], ...]],
      el_type: Type,
  ) -> "Tensor":

    return cls(tensor.EmptyOp(shape, el_type).result)

  def __class_getitem__(
      cls, dim_sizes_dtype: Tuple[Union[list[int], tuple[int, ...]],
                                  Type]) -> Type:
    if len(dim_sizes_dtype) != 2:
      raise ValueError(
          f"Wrong type of argument to {cls.__name__}: {dim_sizes_dtype=}")
    dim_sizes, dtype = dim_sizes_dtype
    if not isinstance(dtype, Type):
      raise ValueError(f"{dtype=} is not {Type=}")
    static_sizes = []
    for s in dim_sizes:
      if isinstance(s, int):
        static_sizes.append(s)
      else:
        static_sizes.append(ShapedType.get_dynamic_size())
    return RankedTensorType.get(static_sizes, dtype)

  def __getitem__(self, idx) -> Scalar:
    if isinstance(idx, (slice, Tensor)):
      return build_gather(self, (idx,))
    else:
      # TODO(max): extract doesn't work because ellipses aren't filled in
      # i.e., move this to index_to_gather
      # idx = tuple(idx)
      # if all(isinstance(i, int) for i in idx):
      #   idx = [arith.ConstantOp.create_index(i).result for i in idx]
      #   return Scalar(tensor.ExtractOp(self, idx))
      # else:
      idx = [Scalar(i, index=True) if isinstance(i, int) else i for i in idx]
      return build_gather(self, tuple(idx))


#################
# index_to_gather
#################


@register_attribute_builder("Arith_CmpIPredicateAttr")
def _predicateAttr(predicate, context):
  # equal (mnemonic: `"eq"`; integer value: `0`)
  # not equal (mnemonic: `"ne"`; integer value: `1`)
  # signed less than (mnemonic: `"slt"`; integer value: `2`)
  # signed less than or equal (mnemonic: `"sle"`; integer value: `3`)
  # signed greater than (mnemonic: `"sgt"`; integer value: `4`)
  # signed greater than or equal (mnemonic: `"sge"`; integer value: `5`)
  # unsigned less than (mnemonic: `"ult"`; integer value: `6`)
  # unsigned less than or equal (mnemonic: `"ule"`; integer value: `7`)
  # unsigned greater than (mnemonic: `"ugt"`; integer value: `8`)
  # unsigned greater than or equal (mnemonic: `"uge"`; integer value: `9`)
  predicates = {
      "eq": 0,
      "ne": 1,
      "slt": 2,
      "sle": 3,
      "sgt": 4,
      "sge": 5,
      "ult": 6,
      "ule": 7,
      "ugt": 8,
      "uge": 9,
  }
  if predicate not in predicates:
    raise ValueError(f"predicate {predicate} not in predicates.")
  predicate = IntegerAttr.get(IntegerType.get_signless(64),
                              predicates[predicate])
  return predicate


def _reshape(arr, shape):
  if isinstance(arr.owner.opview, arith.ConstantOp):
    np_arr = np.array(DenseElementsAttr(arr.owner.opview.value)).reshape(shape)
    return _as_index_tensor(np_arr)
  else:
    return Tensor(
        tensor.ReshapeOp(RankedTensorType.get(shape, arr.dtype), arr,
                         _as_index_tensor(shape)))


def _is_empty_shape(shape):
  if isinstance(shape, (tuple, list)):
    if not all(isinstance(s, int) for s in shape):
      raise NotImplementedError(shape)

    return any(s == 0 for s in shape)
  raise NotImplementedError(shape)


class _Indexer(NamedTuple):
  indices: Tensor
  # The set of dimensions i in operand that have slice_sizes[i] == 1.
  # Must be a tuple of integers in ascending order.
  # These are either gather or scatter dims.
  collapsed_dims: Tuple[int, ...]

  # Keep track of any axes created by `newaxis`. These must be inserted for
  # gathers and eliminated for scatters.
  newaxis_dims: Sequence[int]

  # Are the gather_indices known to be non-overlapping and/or sorted?
  unique_indices: Optional[bool] = None


def _concatenate(arrs: Sequence[Tensor], dim):
  assert len(arrs) > 1
  if all([a.is_constant() for a in arrs]):
    return Tensor(np.concatenate([a.literal_value for a in arrs], axis=dim),
                  index=True)
  else:
    return Tensor(ConcatenateOp(arrs, dim))


def _zeros(shape, dtype):
  if isinstance(shape, tuple):
    shape = list(shape)
  typ = RankedTensorType.get(shape, dtype)
  if (F16Type.isinstance(dtype) or F32Type.isinstance(dtype) or
      F64Type.isinstance(dtype)):
    attr = FloatAttr.get(dtype, 0.0)
  else:
    attr = IntegerAttr.get(dtype, 0)

  return Tensor(arith.ConstantOp(typ, DenseElementsAttr.get_splat(typ, attr)))


def _arange(start, limit, stride, dtype, expand=False):
  if all(isinstance(e, int)
         for e in (start, limit, stride)) and dtype == IndexType:
    arr = np.arange(start, limit, stride)
    if expand:
      arr = arr[:, np.newaxis]
    return _as_index_tensor(arr)
  raise NotImplementedError(start, limit, stride, dtype)


def _static_idx(idx: slice, size):
  """Helper function to compute the static slice start/limit/stride values."""
  if isinstance(size, int):
    start, stop, step = idx.indices(size)
  else:
    raise TypeError(size)

  if (step < 0 and stop >= start) or (step > 0 and start >= stop):
    return 0, 0, 1, False  # sliced to size zero

  if step > 0:
    return start, stop, step, False
  else:
    k = (start - stop - 1) % (-step)
    return stop + k + 1, start + 1, -step, True


def _is_slice_element_none_or_constant(elt):
  if (elt is None or isinstance(elt, int) or isinstance(elt, Value) and
      isinstance(elt.owner.opview, arith.ConstantOp)):
    return True
  raise NotImplementedError(elt)


def _dimension_as_value(axis_size):
  raise NotImplementedError(axis_size)


def _greater_equal_dim(lhs, rhs):
  if isinstance(lhs, int) and isinstance(rhs, int):
    return lhs >= rhs

  raise NotImplementedError(lhs, rhs)


def _expand_dims(y, newaxis_dims):
  if len(newaxis_dims) == 0:
    return y
  if isinstance(y, Scalar):
    assert newaxis_dims == (0,), f"{newaxis_dims=}"
    if y.is_constant():
      return _as_index_tensor(np.array(y.literal_value).reshape((1,)))
    else:
      return Tensor(
          tensor.FromElementsOp(RankedTensorType.get((1,), y.type), [y]))
  elif Tensor.isinstance(y):
    if len(set(newaxis_dims)) != len(newaxis_dims):
      raise ValueError(f"repeated axis in expand_dims: {newaxis_dims}")
    ndim_out = len(y.shape) + len(newaxis_dims)
    assert all(
        0 <= d < ndim_out for d in newaxis_dims), "no negative dims allowed"
    dims_set = frozenset(newaxis_dims)
    reassoc_list = [[i] for i in range(len(y.shape))]
    result_shape = list(y.shape)
    # TODO(max): check this logic
    for i in sorted(dims_set):
      reassoc_list[i - 1].append(i)
      result_shape.insert(i, 1)
    if y.is_constant():
      return _as_index_tensor(y.literal_value)
    else:
      reassoc_list = _get_int_int_array_attr(reassoc_list)
      return Tensor(
          tensor.ExpandShapeOp(RankedTensorType.get(result_shape, y.dtype), y,
                               reassoc_list))

  raise NotImplementedError(y, newaxis_dims)


def _dtype(e):
  if isinstance(e, Scalar):
    return e.dtype
  elif isinstance(e, (int, float, bool)):
    return type(e)
  else:
    raise NotImplementedError(e)


def _const(example, val):
  return Scalar(arith.ConstantOp(_dtype(example), val))


def _is_constant_dim(d):
  if isinstance(d, int):
    return True
  raise NotImplementedError(d)


def _normalize_index(index, axis_size):
  """Normalizes an index value in the range [-N, N) to the range [0, N)."""
  if isinstance(index.owner.opview, arith.ConstantOp):
    if 0 <= index.owner.opview.literal_value < axis_size:
      return index
  if _is_constant_dim(axis_size):
    axis_size_val = _const(index, axis_size)
  else:
    axis_size_val = _convert_element_type(_dimension_as_value(axis_size),
                                          _dtype(index))
  zero = arith.ConstantOp.create_index(0)
  cond = arith.CmpIOp("ult", index, zero)
  true = arith.AddIOp(index, axis_size_val)
  return Scalar(arith.SelectOp(cond, true, index))


def _symbolic_equal_dim(param, param1):
  if isinstance(param, int) and isinstance(param1, int):
    return param == param1
  raise NotImplementedError(param, param1)


def _convert_element_type(a, index_dtype):
  if isinstance(a, Value) and index_dtype.isinstance(a.type):
    return a
  if isinstance(a, int):
    return _as_index_tensor([a])

  raise NotImplementedError(a, index_dtype)


def _diff(idx_advanced_axes):
  if len(idx_advanced_axes) == 1:
    return []
  else:
    return [
        idx_advanced_axes[i + 1] - idx_advanced_axes[i]
        for i in range(0,
                       len(idx_advanced_axes) - 1, 2)
    ]


def _as_index_tensor(e):
  if isinstance(e, Value) and Tensor.isinstance(e):
    return e

  return Tensor(np.array(e), index=True)


def _is_index_tensor(x):
  return (isinstance(x, Value) and Tensor.isinstance(x) and
          IndexType.isinstance(x.dtype))


def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise."""
  return (isinstance(x, int) and not isinstance(x, bool) or
          isinstance(x, (list, tuple)) and all(_is_int_arraylike(e) for e in x))


def _has_index_type(e):
  return isinstance(e, int) or IndexType.isinstance(e.type)


def _is_scalar(e):
  return isinstance(e, Scalar) or isinstance(e, (int, float, bool))


def _is_advanced_int_indexer(idx):
  assert isinstance(idx, tuple)
  if all(e is None or e is Ellipsis or isinstance(e, slice) or
         _is_scalar(e) and _has_index_type(e) for e in idx):
    return False
  return all(e is None or e is Ellipsis or isinstance(e, slice) or
             _is_int_arraylike(e) or _is_index_tensor(e) for e in idx)


def _canonicalize_tuple_index(arr_ndim, idx, array_name="array"):
  """Helper to remove Ellipsis and add in the implicit trailing slice(None)."""
  if _is_advanced_int_indexer(idx):
    idx = list(idx)
    for idx_i, idx_e in enumerate(idx):
      if idx_e is not None and not isinstance(idx_e,
                                              slice) and idx_e != Ellipsis:
        idx[idx_i] = _as_index_tensor(idx_e)
    idx = tuple(idx)

  len_without_none = 0
  for e in idx:
    if e is None or e is Ellipsis:
      continue
    if _is_index_tensor(e):
      len_without_none += e.shape[-1]
    else:
      len_without_none += 1

  if len_without_none > arr_ndim:
    raise IndexError(f"Too many indices for {array_name}: {len_without_none} "
                     f"non-None/Ellipsis indices for dim {arr_ndim}.")
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      raise IndexError(
          f"Multiple ellipses (...) not supported: {list(map(type, idx))}.")
    colons = (slice(None),) * (arr_ndim - len_without_none)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif len_without_none < arr_ndim:
    colons = (slice(None),) * (arr_ndim - len_without_none)
    idx = tuple(idx) + colons
  return idx


def _index_to_scatter_gather(x_shape: Sequence[int],
                             idx: Sequence[Any],
                             normalize_indices: bool = True):
  # Remove ellipses and add trailing slice(None)s.
  idx = _canonicalize_tuple_index(len(x_shape), idx)

  # Do the advanced indexing axes appear contiguously? If not, NumPy semantics
  # move the advanced axes to the front.
  advanced_axes_are_contiguous = False
  advanced_indexes: Optional[Sequence[Union[Tensor, Sequence]]] = None

  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes: Sequence[int] = []

  if _is_advanced_int_indexer(idx):
    idx_no_nones = [(i, d) for i, d in enumerate(idx) if d is not None]
    advanced_pairs = []
    for idx_i, idx_e in idx_no_nones:
      if _is_index_tensor(idx_e):
        advanced_pairs.append((idx_e, idx_i))

    advanced_indexes, idx_advanced_axes = zip(*advanced_pairs)
    advanced_axes_are_contiguous = bool(
        all([d == 1 for d in _diff(idx_advanced_axes)]))

  in_axis = 0  # Current axis in input.
  out_axis = 0  # Current axis in output, before collapsing. See below.

  collapsed_dims: Sequence[int] = []

  indices: List[Tensor] = []
  indices_shape: List[int] = []

  newaxis_dims: Sequence[int] = []

  for idx_i, idx_e in enumerate(idx):
    # Handle the advanced indices here if:
    # * the advanced indices were not contiguous, and we are the start.
    # * we are at the position of the first advanced index.
    if advanced_indexes is not None and (
        advanced_axes_are_contiguous and idx_i == idx_advanced_axes[0] or
        not advanced_axes_are_contiguous and idx_i == 0):
      if len(set([tuple(a.shape) for a in advanced_indexes])) != 1:
        raise IndexError("All advanced indices must have the same shape.")
      shape = advanced_indexes[0].shape

      indices.extend(advanced_indexes)
      indices_shape += shape

      prev_idx = collapsed_dims[-1] if len(collapsed_dims) else -1
      for idx_pos, idx_ in enumerate(idx):
        if idx_pos in idx_advanced_axes:
          collapsed_dims.extend(np.array(prev_idx + 1) + np.arange(shape[-1]))
          prev_idx = collapsed_dims[-1]
        else:
          prev_idx += 1

      ndim = len(shape)
      out_axis += ndim

    # Per-index bookkeeping for advanced indexes.
    if idx_i in idx_advanced_axes:
      in_axis += idx[idx_i].shape[-1]
      continue

    # Handle basic int indexes.
    if _is_scalar(idx_e) and _has_index_type(idx_e):
      if _symbolic_equal_dim(x_shape[in_axis], 0):
        raise IndexError(
            f"index is out of bounds for axis {in_axis} with size 0")
      idx_e = (_normalize_index(idx_e, x_shape[in_axis])
               if normalize_indices else idx_e)
      idx_e = _convert_element_type(idx_e, IndexType)
      idx_e = _expand_dims(idx_e, (0,))
      indices.append(idx_e)
      collapsed_dims.append(in_axis)
      in_axis += 1
    # Handle newaxis (None)
    elif idx_e is None:
      newaxis_dims.append(out_axis)
      out_axis += 1

    elif isinstance(idx_e, slice):
      # Normalize the slice to use None when possible
      start, stop, step = idx_e.start, idx_e.stop, idx_e.step
      if step is None or _symbolic_equal_dim(step, 1):
        step = None
      if step is None:
        if start is None or _symbolic_equal_dim(start, 0):
          start = None
        if stop is None or _greater_equal_dim(stop, x_shape[in_axis]):
          stop = None
      elif _symbolic_equal_dim(step, -1):
        step = -1

      # Handle slice(None) and slice(None, None, -1)
      if (start is None and stop is None and
          (step is None or isinstance(step, int) and step == -1)):
        if step == -1:
          raise NotImplementedError(step)
        out_axis += 1
        in_axis += 1
      # Handle slice index (only static, otherwise an error is raised)
      else:
        if not all(
            _is_slice_element_none_or_constant(elt)
            for elt in (start, stop, step)):
          msg = (
              "Array slice indices must have static start/stop/step to be used "
              "with NumPy indexing syntax. "
              f"Found slice({start}, {stop}, {step}). ")
          raise IndexError(msg)
        if not _is_constant_dim(x_shape[in_axis]):
          msg = ("Cannot use NumPy slice indexing on an array dimension whose "
                 f"size is not statically known ({x_shape[in_axis]}). ")
          raise IndexError(msg)
        start, limit, stride, needs_rev = _static_idx(slice(start, stop, step),
                                                      x_shape[in_axis])
        if needs_rev:
          raise NotImplementedError(needs_rev)

        idx_e = _arange(start, limit, stride, dtype=IndexType, expand=True)
        size = idx_e.shape[0]
        indices.append(idx_e)
        indices_shape.append(size)
        collapsed_dims.append(in_axis)

        out_axis += 1
        in_axis += 1
    else:
      raise IndexError(
          f"Indexing mode not yet supported. Open a feature request!\n{idx}")

  collapsed_dims = tuple(sorted(collapsed_dims))

  if len(indices) == 0:
    # empty array
    indices_array = _zeros((0,), dtype=IndexType.get())
  elif len(indices) == 1:
    indices_array = indices[0]
  else:
    if len(indices_shape) == 0:
      last_dim = 0
    else:
      last_dim = len(indices_shape) - 1
    indices_array = _concatenate(
        indices,
        last_dim,
    )

  if indices_array.is_constant():
    lit = indices_array.literal_value
    # flatten all but last dim (i.e., idx/coord dim)
    coords = lit.reshape(-1, lit.shape[-1])
    unique_indices = len(np.unique(coords, axis=0)) == len(coords)
  else:
    unique_indices = advanced_indexes is None

  return _Indexer(
      newaxis_dims=tuple(newaxis_dims),
      collapsed_dims=collapsed_dims,
      indices=indices_array,
      unique_indices=unique_indices,
  )


def gather(
    operand,
    indices,
    gather_dims,
    *,
    unique_indices: bool = False,
):
  return Tensor(
      GatherOp(
          source=operand,
          indices=indices,
          gather_dims=gather_dims,
          unique=unique_indices,
      ))


def build_gather(
    arr,
    idx,
    unique_indices=False,
):
  # early bail for most trivial corner case (all full slices)
  if all(i == slice(None) for i in idx):
    return arr

  indexer = _index_to_scatter_gather(arr.shape, idx)
  y = arr

  # We avoid generating a gather when indexer.gather_indices.size is empty.
  if not _is_empty_shape(indexer.indices.shape):
    y = gather(
        y,
        indexer.indices,
        indexer.collapsed_dims,
        unique_indices=unique_indices or indexer.unique_indices,
    )

  # This adds newaxis/None dimensions.
  return _expand_dims(y, indexer.newaxis_dims)
