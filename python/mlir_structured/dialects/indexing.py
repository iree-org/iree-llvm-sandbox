# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import operator
from copy import deepcopy
from functools import cached_property, lru_cache, partialmethod
from typing import Optional, Tuple, Union, NamedTuple, Sequence, Any, List

import numpy as np

from . import arith, tensor
from ._arith_ops_ext import _is_integer_like_type
from ._indexing_ops_gen import *
from ._ods_common import get_op_result_or_value
from .linalg.opdsl.lang.emitter import _is_floating_point_type
from .._mlir_libs._structuredDialects.indexing import *
from ..ir import (
    DenseElementsAttr,
    F16Type,
    F32Type,
    F64Type,
    FloatAttr,
    IndexType,
    IntegerAttr,
    IntegerType,
    OpView,
    Operation,
    RankedTensorType,
    ShapedType,
    Type,
    Value,
    register_attribute_builder,
)


def infer_mlir_type(
    py_val: Union[int, float, bool, np.ndarray]
) -> Union[IntegerType, F64Type, RankedTensorType]:
  """Infer MLIR type (`ir.Type`) from supported python values.

  Note ints and floats are mapped to 64-bit types.

  Args:
    py_val: Python value that's either a numerical value or numpy array.

  Returns:
    MLIR type corresponding to py_val.
  """
  if isinstance(py_val, bool):
    return IntegerType.get_signless(1)
  elif isinstance(py_val, int):
    return IntegerType.get_signless(64)
  elif isinstance(py_val, float):
    return F64Type.get()
  elif isinstance(py_val, np.ndarray):
    dtype = {
        np.int8: IntegerType.get_signless(8),
        np.int16: IntegerType.get_signless(16),
        np.int32: IntegerType.get_signless(32),
        np.int64: IntegerType.get_signless(64),
        np.uintp: IndexType.get(),
        np.longlong: IndexType.get(),
        np.float16: F16Type.get(),
        np.float32: F32Type.get(),
        np.float64: F64Type.get(),
    }[py_val.dtype.type]
    return RankedTensorType.get(py_val.shape, dtype)
  else:
    raise NotImplementedError(
        f"Unsupported Python value {py_val=} with type {type(py_val)}")


def constant(
    value: Union[int, float, bool, np.ndarray],
    type: Optional[Type] = None,
    index: Optional[bool] = None,
) -> arith.ConstantOp:
  """Instantiate arith.constant with value `value`.

  Args:
    value: Python value that determines the value attribute of the
      arith.constant op.
    type: Optional MLIR type that type of the value attribute of the
      arith.constant op; if omitted the type of the value attribute
      will be inferred from the value.
    index: Whether the MLIR type should be an index type; if passed the
      type argument will be ignored.

  Returns:
    ir.OpView instance that corresponds to instantiated arith.constant op.
  """
  if index is not None and index:
    type = IndexType.get()
  if type is None:
    type = infer_mlir_type(value)
  assert type is not None

  if isinstance(value, np.ndarray):
    value = DenseElementsAttr.get(
        value,
        type=type,
    )

  return arith.ConstantOp(type, value)


class ArithValueMeta(type(Value)):
  """Metaclass that orchestrates the Python object protocol
  (i.e., calling __new__ and __init__) for Indexing dialect extension values
  (created using `mlir_value_subclass`).

  The purpose/benefit of handling the `__new__` and `__init__` calls
  explicitly/manually is we can then wrap arbitrary Python objects; e.g.
  all three of the following wrappers are equivalent:

  ```
  s1 = Scalar(arith.ConstantOp(f64, 0.0).result)
  s2 = Scalar(arith.ConstantOp(f64, 0.0))
  s3 = Scalar(0.0)
  ```

  In general the Python object protocol for an object instance is determined
  by `__call__` of the object class's metaclass, thus here we overload
  `__call__` and branch on what we're wrapping there.

  Why not just overload __new__ and be done with it? Because then we can't
  choose what get's passed to __init__: by default (i.e., without overloading
  __call__ here) the same arguments are passed to both __new__ and __init__.

  Note, this class inherits from `type(Value)` (i.e., the metaclass of
  `ir.Value`) rather than `type` or `abc.ABCMeta` or something like this because
  the metaclass of a derived class must be a (non-strict) subclass of the
  metaclasses of all its bases and so all the extension classes
  (`ScalarValue`, `TensorValue`), which are derived classes of `ir.Value` must
  have metaclasses that inherit from the metaclass of `ir.Value`. Without this
  hierarchy Python will throw `TypeError: metaclass conflict`.
  """

  def __call__(cls, *args, **kwargs):
    """Orchestrate the Python object protocol for Indexing dialect extension
    values in order to handle wrapper arbitrary Python objects.

    Args:
      *args: Position arguments to the class constructor. Note, currently,
        only one positional arg is supported (so constructing something like a
        tuple type from element objects isn't supported).
      **kwargs: Keyword arguments to the class constructor. Note, currently,
        we only look for `dtype` (an `ir.Type`)  and `fold`, which determines
        whether binary operations on constant instances will be
        folded (i.e., evaluated at runtime).

    Returns:
      A fully constructed and initialized instance of the class.
    """
    if len(args) != 1:
      raise ValueError("Only one non-kw arg supported.")
    arg = args[0]
    arg_copy = None
    fold = None
    if isinstance(arg, (OpView, Operation, Value)):
      # wrap an already created Value (or op the produces a Value)
      if isinstance(arg, (Operation, OpView)):
        assert len(arg.results) == 1
      val = get_op_result_or_value(arg)
    elif isinstance(arg, (int, float, bool, np.ndarray)):
      # wrap a Python value, effectively a scalar or tensor literal
      dtype = kwargs.get("dtype")
      if dtype is not None and not isinstance(dtype, Type):
        raise ValueError(f"{dtype=} is expected to be an ir.Type.")
      fold = kwargs.get("fold")
      if fold is not None and not isinstance(fold, bool):
        raise ValueError(f"{fold=} is expected to be a bool.")

      # If we're wrapping a numpy array (effectively a tensor literal),
      # then we want to make sure no one else has access to that memory.
      # Otherwise, the array will get funneled down to DenseElementsAttr.get,
      # which by default (through the Python buffer protocol) does not copy;
      # see mlir/lib/Bindings/Python/IRAttributes.cpp#L556
      arg_copy = deepcopy(arg)
      val = constant(arg, dtype).result
    else:
      raise NotImplementedError(
          f"{cls.__name__} doesn't support wrapping {arg}.")

    # The mlir_value_subclass mechanism works through __new__
    # (see mlir/Bindings/Python/PybindAdaptors.h#L502)
    # So we have to pass the wrapped Value to the __new__ of the subclass
    cls_obj = cls.__new__(cls, val)
    # We also have to pass it to __init__ because that is required by
    # the Python object protocol; first an object is new'ed and then
    # it is init'ed. Note we pass arg_copy here in case a subclass wants to
    # inspect the literal.
    cls.__init__(cls_obj, val, arg_copy, fold=fold)
    return cls_obj


class ArithValue(metaclass=ArithValueMeta):
  """Mixin class for functionality shared by mlir_value_subclasses that support
  arithmetic operations.

  Note, since we bind the ArithValueMeta here, it is here that the __new__ and
  __init__ must be defined. To be precise, the callchain, starting from
  ArithValueMeta is:

  ArithValueMeta.__call__ -> mlir_value_subclass.__new__ ->
                        (mlir_value_subclass.__init__ == ArithValue.__init__) ->
                        Value.__init__
  """

  def __init__(
      self,
      val,
      arg: Optional[Union[int, float, bool, np.ndarray]] = None,
      fold: Optional[bool] = None,
  ):
    self.__arg = arg
    self._fold = fold if fold is not None else True
    super().__init__(val)

  @lru_cache(maxsize=1)
  def is_constant(self) -> bool:
    return isinstance(self.owner.opview, arith.ConstantOp)

  @lru_cache(maxsize=1)
  def fold(self) -> bool:
    return self.is_constant() and self._fold

  @lru_cache(maxsize=1)
  def __str__(self):
    if self.is_constant():
      v = str(self.literal_value)
      v = f"{v[:10]}..." if len(v) > 10 else v
      return f"{self.__class__.__name__}({self.get_name()}, {self.type}, {v})"
    else:
      return f"{self.__class__.__name__}({self.get_name()}, {self.type})"

  @lru_cache(maxsize=1)
  def __repr__(self):
    return str(self)

  def __binary_op(self, other: "ArithValue", op: str) -> "ArithValue":
    """Generic for handling infix binary operator dispatch.

    Args:
      self: E.g. Scalar or Tensor below.
      other: Scalar or Tensor with type matching self.
      op: Binary operator, currently only add, sub, mul
        supported.

    Returns:
      Result of binary operation. If constant folding is possible and
      performed then this will be a handle to an arith.constant op and
      otherwise to an arith(add|sub|mul) op.
    """
    assert op in {"add", "sub", "mul"}
    if self.type != other.type:
      raise ValueError(f"{self=} {other=} must have the same type.")

    if self.fold() and other.fold():
      # if both operands are constants (results of an arith.constant op)
      # then both have a literal value (i.e. Python value).
      lhs, rhs = self.literal_value, other.literal_value
      # if we're folding constants (self._fold = True) then we just carry out
      # the corresponding operation on the literal values; e.g., operator.add.
      # note this is the same as op = operator.__dict__[op].
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

  # partialmethod differs from partial in that it also binds the object instance
  # to the first arg (i.e., self)
  __add__ = partialmethod(__binary_op, op="add")
  __sub__ = partialmethod(__binary_op, op="sub")
  __mul__ = partialmethod(__binary_op, op="mul")


class Scalar(ArithValue, ScalarValue):
  """Decorator for mlir_value_subclass ScalarValue that adds convenience methods
  for getting dtype and (possibly) the stored literal value.

  Note, order matters in the superclasses above; ArithValue is first so that
  e.g. __init__, and __str__ from ArithValue are used instead of
  from ScalarValue.
  """

  @cached_property
  def dtype(self) -> Type:
    return self.type

  @cached_property
  def literal_value(self) -> Union[int, float, bool]:
    if not self.is_constant():
      raise ValueError("Can't build literal from non-constant Scalar")
    return self.owner.opview.literal_value


class Tensor(ArithValue, TensorValue):
  """Decorator for mlir_value_subclass TensorValue that adds convenience methods
  for getting dtype, shape and (possibly) the stored literal value.

  Note, order matters in the superclasses above; ArithValue is first so that
  e.g. __init__, and __str__ from ArithValue are used instead of
  from TensorValue.
  """

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
    """A convenience method for creating RankedTensorType.

    Args:
      dim_sizes_dtype: A tuple of both the shape of the type and the dtype.

    Returns:
      An instance of RankedTensorType.
    """
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

  def __getitem__(self, idx: tuple) -> Scalar:
    # Early bail for most trivial corner cases (ellipse or one slice(None))
    if idx == Ellipsis or idx == slice(None):
      return self
    idx = list((idx,) if isinstance(idx, int) else idx)
    for i, d in enumerate(idx):
      if isinstance(d, int):
        idx[i] = Scalar(arith.ConstantOp.create_index(d))

    if all(isinstance(d, Scalar) and d.is_constant()
           for d in idx) and len(idx) == len(self.shape):
      return Scalar(tensor.ExtractOp(self, idx))
    else:
      return build_gather(self, tuple(idx))


###################
# advanced indexing
###################


class _Indexer(NamedTuple):
  """Data that describes/captures the relationship between an indexing tensor
  (produced from simple or advanced indices) and scatter/gather dims.

  Args:
    indices: The final index tensor.
    collapsed_dims: The set of dimensions i in operand that have
      shape[i] == 1. Must be a tuple of integers in ascending order.
      These determine the gather or scatter dims.
    newaxis_dims: axes created by `np.newaxis` or None. These must be inserted
      for gathers and eliminated for scatters.
    unique_indices: Describes whether indices are known to be non-overlapping.
  """

  indices: Tensor
  collapsed_dims: Tuple[int, ...]
  newaxis_dims: Tuple[int, ...]
  unique_indices: Optional[bool] = None


def _is_empty(ten: Tensor) -> bool:
  """Check whether any of the dimensions of `ten` are degenerate (i.e., have
  size 0).

  Args:
    ten: Tensor whose shape is under consideration.

  Returns:
    Whether any of the dimensions of `ten` are degenerate.
  """
  shape = ten.shape
  if isinstance(shape,
                (tuple, list)) and all(isinstance(s, int) for s in shape):
    return any(s == 0 for s in shape)
  raise NotImplementedError(shape)


def concatenate(tens: Sequence[Tensor], dim) -> Tensor:
  """Concatenate a sequence Tensors along dimension `dim`.

  Args:
    tens: Sequence of tensors all having the same shape, except for along
      dimension `dim`.
    dim: Dimension to concatenate along.

  Returns:
    Tensor that wraps a value that's either the result of index.concatenate
    or (if all tensors are constant and foldable) the result of arith.constant.
  """
  if all(a.fold() for a in tens):
    return Tensor(np.concatenate([a.literal_value for a in tens], axis=dim))
  else:
    return Tensor(ConcatenateOp(tens, dim))


def _as_index_tensor(val) -> Tensor:
  """Cast to index tensor.

  Args:
    val: Python value that an index tensor can be constructed from.

  Returns:
    Tensor with index element type.
  """
  return Tensor(np.array(val), dtype=IndexType.get())


def _expand_dims(y, axis: int) -> Tensor:
  """Expand the shape of a tensor.

  Insert a new axis that will appear at the `axis` position in the expanded
  tensor shape.

  Args:
    y: Input tensor-like.
    axis: Position in the expanded axes where the new axis (or axes) is placed.

  Returns:
     View of `a` with the number of dimensions increased.

  """
  if len(axis) == 0:
    return y
  if isinstance(y, Scalar):
    assert axis == (0,), f"Expected axis to be 0 but {axis=}."
    if y.fold():
      return Tensor(np.array(y.literal_value).reshape((1,)), dtype=y.dtype)

  raise NotImplementedError(y, axis)


def _has_index_type(e: Any) -> bool:
  """Checks whether e has MLIR index type or a Python value that can be used
  to construct an index type.

  Args:
    e: Anything
  """
  return isinstance(e, int) or isinstance(e, np.ndarray) and e.dtype in {
      np.uintp, np.longlong
  } or isinstance(e, (Tensor, Scalar)) and IndexType.isinstance(e.dtype)


def _is_scalar(e: Any) -> bool:
  """Checks whether e is a Scalar or can be used to construct a Scalar.

  Args:
    e: Anything
  """
  return isinstance(e, Scalar) or isinstance(e, (int, float, bool))


def _canonicalize_tuple_index(idx, rank: int):
  """Helper to remove Ellipsis and replace with implicit trailing slice(None)s.

  Args:
    rank: Rank of tensor.
    idx: Index object (Scalar, Tensor, slice, Ellipse, or None).

  Returns:
    Tuple of index objects with no ellipses.
  """
  len_without_none = 0
  for e in idx:
    if e is None or e is Ellipsis:
      continue
    else:
      len_without_none += 1

  if len_without_none > rank:
    raise IndexError(f"Too many indices for tensor: {len_without_none} "
                     f"non-None/Ellipsis indices for dim {rank}.")
  ellipses = (i for i, elt in enumerate(idx) if elt is Ellipsis)
  ellipsis_index = next(ellipses, None)
  if ellipsis_index is not None:
    if next(ellipses, None) is not None:
      raise IndexError(
          f"Multiple ellipses (...) not supported: {list(map(type, idx))}.")
    colons = (slice(None),) * (rank - len_without_none)
    idx = idx[:ellipsis_index] + colons + idx[ellipsis_index + 1:]
  elif len_without_none < rank:
    colons = (slice(None),) * (rank - len_without_none)
    idx = tuple(idx) + colons
  return idx


def _indices_to_indexer(idx: Sequence[Any],
                        in_shape: Sequence[int]) -> _Indexer:
  """Processes sequence of index objects and constructs _Indexer with
  corresponding indexing tensor and collapse dims (i.e., scatter/gather dims).

  Args:
    idx: Sequence (list or tuple) of slices, ellipses, Scalar, or Tensors.
    in_shape: The shape of the tensor being indexed into.

  Returns:
    _Indexer object.

  """
  idx = _canonicalize_tuple_index(idx, len(in_shape))

  in_axis = 0  # Current axis in input.
  out_axis = 0  # Current axis in output, before collapsing. See below.
  collapsed_dims: Sequence[int] = []
  indices: List[Tensor] = []
  newaxis_dims: Sequence[int] = []

  for idx_i, idx_e in enumerate(idx):
    if _is_scalar(idx_e) and _has_index_type(idx_e):
      # Handle basic int indexes.
      idx_e = _expand_dims(idx_e, (0,))
      indices.append(idx_e)
      collapsed_dims.append(in_axis)
      in_axis += 1
    elif isinstance(idx_e, slice):
      # Handle slice indices
      out_axis += 1
      in_axis += 1
    else:
      raise IndexError(
          f"Indexing mode not yet supported. Open a feature request!\n{idx}")

  collapsed_dims: Tuple[int, ...] = tuple(sorted(collapsed_dims))

  if len(indices) == 1:
    indices_tensor = indices[0]
  else:
    indices_tensor = concatenate(
        indices,
        0,
    )

  lit = indices_tensor.literal_value
  # flatten all but last dim (i.e., idx/coord dim)
  coords = lit.reshape(-1, lit.shape[-1])
  unique_indices = len(np.unique(coords, axis=0)) == len(coords)

  return _Indexer(
      newaxis_dims=tuple(newaxis_dims),
      collapsed_dims=collapsed_dims,
      indices=indices_tensor,
      unique_indices=unique_indices,
  )


def gather(
    operand: Tensor,
    indices: Tensor,
    gather_dims: Tuple[int, ...],
    *,
    unique_indices: bool = False,
) -> Tensor:
  return Tensor(
      GatherOp(
          source=operand,
          indices=indices,
          gather_dims=gather_dims,
          unique=unique_indices,
      ))


def build_gather(
    ten: Tensor,
    idx,
    unique_indices=False,
) -> Tensor:
  # Early bail for most trivial corner case (all full slices)
  if all(i == slice(None) for i in idx):
    return ten

  indexer = _indices_to_indexer(idx, ten.shape)
  out = ten

  # We avoid generating a gather when indexer.indices is empty i.e., has any
  # zero dims.
  if _is_empty(indexer.indices):
    return out

  return gather(
      out,
      indexer.indices,
      indexer.collapsed_dims,
      unique_indices=unique_indices or indexer.unique_indices,
  )
