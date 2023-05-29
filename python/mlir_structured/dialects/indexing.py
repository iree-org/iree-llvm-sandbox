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
from ._structured_transform_ops_ext import _get_int_int_array_attr
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

NP_DTYPE_TO_MLIR_TYPE = lambda: {
    np.int8:
        IntegerType.get_signless(8),
    np.int16:
        IntegerType.get_signless(16),
    np.int32:
        IntegerType.get_signless(32),
    np.int64:
        IntegerType.get_signless(64),
    # this is techincally wrong i guess but numpy by default casts python scalars to this
    # so to support passing lists of ints we map this to index type
    np.longlong:
        IndexType.get(),
    np.uintp:
        IndexType.get(),
    np.float16:
        F16Type.get(),
    np.float32:
        F32Type.get(),
    np.float64:
        F64Type.get(),
}

MLIR_TYPE_TO_NP_DTYPE = lambda: {
    v: k for k, v in NP_DTYPE_TO_MLIR_TYPE().items()
}


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
    dtype = NP_DTYPE_TO_MLIR_TYPE()[py_val.dtype.type]
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
  elif RankedTensorType.isinstance(type) and isinstance(value,
                                                        (int, float, bool)):
    ranked_tensor_type = RankedTensorType(type)
    value = np.ones(
        ranked_tensor_type.shape,
        dtype=MLIR_TYPE_TO_NP_DTYPE()[ranked_tensor_type.element_type])
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
    if not isinstance(other, self.__class__):
      other = self.__class__(other, dtype=self.type)

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
    return self.__class__(op(lhs, rhs), dtype=self.dtype)

  # partialmethod differs from partial in that it also binds the object instance
  # to the first arg (i.e., self)
  __add__ = partialmethod(__binary_op, op="add")
  __sub__ = partialmethod(__binary_op, op="sub")
  __mul__ = partialmethod(__binary_op, op="mul")
  __radd__ = partialmethod(__binary_op, op="add")
  __rsub__ = partialmethod(__binary_op, op="sub")
  __rmul__ = partialmethod(__binary_op, op="mul")


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
    """Advanced indexing for tensors.

    This method implements (primarily) a mapping from Numpy style advanced indexing
    (see https://numpy.org/doc/stable/user/basics.indexing.html#advanced-indexing)
    to indexing.GatherOp. In particular that means element indexing like ten[0, 1, 2, 3],
    indexing with slicing like ten[0, :, 1, :], and indexing with lists and tensors is mapped
    to IR like
        indexing.gather %[ten][...%idx_tensor...] gather_dims([0]) unique
          : (tensor<10x10x10x10xf32>, tensor<2x1xindex>) -> tensor<2x10x10x10xf32>

    One particular case to emphasize: indexing with multiple advanced indexors (tensors or lists)
    For example, ten[ [[0], [0]] , [[1], [1]] , :, :] where the two separate advanced indexers are
    [[0], [0]] and [[1], [1]]. In such cases (and in more advanced cases) the indexers are concatenated along
    the last dimension to construct the actual coordinate tuples; i.e., in this case the final advanced indexer
    is [[0, 0], [1, 1]], which means the slices selected from the tensor `ten` are ten[0, 0, :, :] and ten[1, 1, :, :].
    The IR generated is like

        indexing.gather %ten[%idx_gensor] gather_dims([0, 1]) unique
          : (tensor<10x10x10x10xf32>, tensor<2x2xindex>) -> tensor<2x10x10xf32>

    For non-contiguous advanced indexers (i.e., indexing objects inserted at non-contiguous positions in the
    indexing tuple), e.g., ten[ [[0], [0]] , :, [[1], [1]] , :], exactly the same things occur and the only thing
    that changes is gather_dims reflect the alternate dimensions:

        indexing.gather %ten[%idx_gensor] gather_dims([0, 2]) unique
          : (tensor<10x10x10x10xf32>, tensor<2x2xindex>) -> tensor<2x10x10xf32>

    i.e., this is a mechanism for expressing non-contiguous gather_dims.

    Args:
      idx: The indexing tuple made up various combinations of integers, lists (and lists of) integers, slice objects (:)
           ellipses, and indexing tensors.

    Returns: A handle to the SSA value corresponding to the result of the indexing.GatherOp instruction.
    """
    if idx == Ellipsis or idx == slice(None):
      return self
    if idx is None:
      return _expand_dims(self, (0,))
    idx = list((idx,) if isinstance(idx, int) else idx)
    for i, d in enumerate(idx):
      if isinstance(d, int):
        idx[i] = Scalar(arith.ConstantOp.create_index(d))

    if all(isinstance(d, Scalar) and d.is_constant()
           for d in idx) and len(idx) == len(self.shape):
      return Scalar(tensor.ExtractOp(self, idx))
    else:
      return _build_gather(self, tuple(idx))


#######################
# advanced indexing api
#######################


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


def arange(start: Union[Scalar, int],
           stop: Optional[Union[Scalar, int]] = None,
           step: Optional[Union[Scalar, int]] = None,
           fold=True):
  """Return evenly spaced values within a given interval.

  * ``arange(stop)``: Values are generated within the half-open interval
    ``[0, stop)`` (in other words, the interval including `start` but
    excluding `stop`).
  * ``arange(start, stop)``: Values are generated within the half-open
    interval ``[start, stop)``.
  * ``arange(start, stop, step)`` Values are generated within the half-open
    interval ``[start, stop)``, with spacing between values given by
    ``step``.

  Args:
    start: Start of interval.  The interval includes this value.  The default
            start value is 0.
    stop: End of interval.  The interval does not include this value.
    step: Spacing between values.  For any output `out`, this is the distance
          between two adjacent values, ``out[i+1] - out[i]``.  The default
          step size is 1.  If `step` is specified as a position argument,
          `start` must also be given.
    fold: Whether operations should be materialized as arrays immediately
          (when possible).

  Returns: Array of evenly spaced values (or handle to operation that
           corresponds).
  """
  index = IndexType.get()
  if isinstance(start, int):
    start = Scalar(start, dtype=index)
  if stop is None and step is None:
    stop = start
    start = 0
    if stop.fold() and fold:
      return Tensor(np.arange(start=start, stop=stop.literal_value,
                              step=1)[:, np.newaxis],
                    dtype=IndexType.get())
    else:
      return Tensor(ARangeOp(start=start, stop=stop, step=1),
                    dtype=IndexType.get())
  else:
    if isinstance(stop, int):
      stop = Scalar(stop, dtype=index)
    if step is None and start.is_constant(
    ) and start.literal_value == 0 and stop is not None:
      if start.fold() and stop.fold() and fold:
        return Tensor(np.arange(start=start.literal_value,
                                stop=stop.literal_value,
                                step=1)[:, np.newaxis],
                      dtype=IndexType.get(),
                      fold=True)
      else:
        return Tensor(ARangeOp(start=start, stop=stop, step=1),
                      dtype=IndexType.get(),
                      fold=start.fold() and stop.fold() and fold)
    if isinstance(step, int):
      step = Scalar(step, dtype=index)
    if start.fold() and stop.fold() and step.fold() and fold:
      return Tensor(np.arange(start=start.literal_value,
                              stop=stop.literal_value,
                              step=step.literal_value)[:, np.newaxis],
                    dtype=IndexType.get(),
                    fold=True)
    else:
      return Tensor(ARangeOp(start=start, stop=stop, step=step),
                    dtype=IndexType.get(),
                    fold=start.fold() and stop.fold() and step.fold() and fold)


########################
# advanced indexing impl
########################


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


def _as_index_tensor(val) -> Tensor:
  """Cast to index tensor.

  Args:
    val: Python value that an index tensor can be constructed from.

  Returns:
    Tensor with index element type.
  """
  return Tensor(np.array(val), dtype=IndexType.get())


def _expand_dims(inp, newaxis_dims) -> Tensor:
  """Expand the shape of a tensor.

  Insert a new axis that will appear at the `axis` position in the expanded
  tensor shape.

  Args:
    inp: Input tensor-like.
    axis: Position in the expanded axes where the new axis (or axes) is placed.

  Returns:
     View of `a` with the number of dimensions increased.

  """
  if len(newaxis_dims) == 0:
    return inp

  newaxis_dims = sorted(newaxis_dims)
  if len(set(newaxis_dims)) != len(newaxis_dims):
    raise ValueError(f"repeated axis in expand_dims: {newaxis_dims}")

  if isinstance(inp, Scalar):
    assert newaxis_dims == [
        0,
    ], f"{newaxis_dims=}"
    if inp.is_constant():
      return _as_index_tensor(np.array(inp.literal_value).reshape((1,)))
    else:
      return Tensor(
          tensor.FromElementsOp(RankedTensorType.get((1,), inp.type), [inp]))
  elif Tensor.isinstance(inp):
    ndim_out = len(inp.shape) + len(newaxis_dims)
    if not all(0 <= d < ndim_out for d in newaxis_dims):
      raise ValueError("no negative dims allowed")
    result_shape = list(inp.shape)
    for i in reversed(newaxis_dims):
      result_shape.insert(i, 1)

    reassoc_list = [[i] for i in range(len(inp.shape))]
    for i, d in enumerate(newaxis_dims):
      reassoc_list.append([len(inp.shape) + i])
      if d == 0:
        d = 1
      reassoc_list[max(d - 1, 0)].extend(reassoc_list.pop(d))

    if inp.is_constant():
      return _as_index_tensor(inp.literal_value)
    else:
      reassoc_list = _get_int_int_array_attr(reassoc_list)
      return Tensor(
          tensor.ExpandShapeOp(RankedTensorType.get(result_shape, inp.dtype),
                               inp, reassoc_list))

  raise NotImplementedError(inp, newaxis_dims)


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


def _is_index_tensor(x):
  """Returns True if x is a Tensor with index dtype, False otherwise."""
  return (isinstance(x, Value) and Tensor.isinstance(x) and
          IndexType.isinstance(x.dtype))


def _is_int_arraylike(x):
  """Returns True if x is array-like with integer dtype, False otherwise.

  Positive (i.e., return True) examples are e.g., [[0], [1]], [[0, 1]],
  [[[0, 1]], [[0, 1]]].
  """
  return (isinstance(x, int) and not isinstance(x, bool) or
          isinstance(x, (list, tuple)) and all(_is_int_arraylike(e) for e in x))


def _is_advanced_int_indexer(idx):
  """Returns True if idx is array-like or slice or Ellipsis or Tensor with index
  dtype, False otherwise.

  Positive (i.e., return True) examples are e.g., [[0], [1]], [[0, 1]],
  [[[0, 1]], [[0, 1]]], [1, 1, 1, :], [1, :, :, 1], [1, :, :, ...],
  [idx, :, :, :], [:, idx, :, :], [:, :, idx, :].

  """
  assert isinstance(idx, tuple)
  if all(e is None or e is Ellipsis or isinstance(e, slice) or
         _is_scalar(e) and _has_index_type(e) for e in idx):
    return False
  return all(e is None or e is Ellipsis or isinstance(e, slice) or
             _is_int_arraylike(e) or _is_index_tensor(e) for e in idx)


def _canonicalize_tuple_index(idx: Tuple[Any], rank: int):
  """Helper to
  1. remove Ellipsis and replace with implicit trailing slice(None)s.
  2. cast Python lists of lists or numpy arrays to index Tensors

  Args:
    rank: Rank of tensor.
    idx: Index object (Scalar, Tensor, slice, Ellipse, or None).

  Returns:
    Tuple of index objects with no ellipses.
  """

  if _is_advanced_int_indexer(idx):
    idx = list(idx)
    for idx_i, idx_e in enumerate(idx):
      if idx_e is not None and not isinstance(
          idx_e, (slice, Tensor)) and idx_e != Ellipsis:
        idx[idx_i] = _as_index_tensor(idx_e)
    idx = tuple(idx)

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


def _compute_idx_tensor_extent(collapsed_dims, idx, idx_advanced_axes):
  """Helper that the proper "extent" (in terms of dims spanned) of an index
  Tensor.

  For example, if ten ~ 10x10x10x10x10 and idx ~ 5x6x2 then
  ten[:, idx, ...] means dims [1, 2] are collapsed (because each coordinate
  tuple in idx has 2 coordinates and idx is indexing starting from the 1st dim).
  While ten[idx, :, idx] means dims [0, 1, 3, 4] are collapsed.
  Args:
    collapsed_dims: Current collapsed dims (this helper is called in a loop
      below that figures out all collapsed dims due to all indexing tensors).
    idx: The total indexing object (for all dims).
    idx_advanced_axes: Axes/dims in original indexing object that are advanced
      indexers.
  Returns: Updated collapsed dims.
  """

  prev_idx = collapsed_dims[-1] if len(collapsed_dims) else -1
  for idx_pos, idx_e in enumerate(idx):
    if idx_pos in idx_advanced_axes:
      # Note it has already been verified that idx_e.shape is the same for all
      # indexers.
      collapsed_dims.extend(np.array(prev_idx + 1) + np.arange(idx_e.shape[-1]))
      prev_idx = collapsed_dims[-1]
    else:
      prev_idx += 1

  return collapsed_dims


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
  out_axis = 0  # Current axis in output.
  collapsed_dims: Sequence[int] = []
  indices: List[Tensor] = []
  indices_shape: List[int] = []
  newaxis_dims: Sequence[int] = []

  # Contiguous in terms of dims that the index/slice
  advanced_axes_are_contiguous = False
  advanced_indexes: Optional[Sequence[Tensor]] = None
  # The positions of the advanced indexing axes in `idx`.
  idx_advanced_axes: Sequence[int] = []

  if _is_advanced_int_indexer(idx):
    advanced_pairs = [(i, d)
                      for i, d in enumerate(idx)
                      if d is not None and _is_index_tensor(d)]
    idx_advanced_axes, advanced_indexes = zip(*advanced_pairs)
    advanced_axes_are_contiguous = bool(
        all([d == 1 for d in np.diff(idx_advanced_axes)]))

  # nb: idx_e <-> idx_element
  for idx_i, idx_e in enumerate(idx):
    # Handle the advanced indices here if:
    # * the advanced indices were not contiguous, and we are the start.
    # * we are at the position of the first advanced index.
    if advanced_indexes is not None and (
        advanced_axes_are_contiguous and idx_i == idx_advanced_axes[0] or
        not advanced_axes_are_contiguous and idx_i == 0):
      if len(set([tuple(a.shape) for a in advanced_indexes])) != 1:
        raise IndexError("All advanced indices must have the same shape.")
      indices.extend(advanced_indexes)
      indices_shape += advanced_indexes[0].shape
      collapsed_dims = _compute_idx_tensor_extent(collapsed_dims, idx,
                                                  idx_advanced_axes)

    if idx_i in idx_advanced_axes:
      # skip ahead the number of coordinates, e.g., for a 5x6x2 idx tensor,
      # skip ahead two axes
      in_axis += idx[idx_i].shape[-1]
      continue

    if _is_scalar(idx_e) and _has_index_type(idx_e):
      # Handle basic Scalar indexes.
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
      if step is None or step == 1:
        step = None
      if step is None:
        if start is None or start == 0:
          start = None
        if stop is None or stop >= in_shape[in_axis]:
          stop = None
      # Handle slice(None) and slice(None, None, -1)
      if (start is None and stop is None and
          (step is None or isinstance(step, int) and step == -1)):
        if step == -1:
          raise IndexError(
              f"Negative step indexing mode not yet supported:\n{idx}")
        out_axis += 1
        in_axis += 1
    else:
      raise IndexError(f"Indexing mode not yet supported:\n{idx}")

  collapsed_dims: Tuple[int, ...] = tuple(sorted(collapsed_dims))

  if len(indices) == 0:
    # empty array
    indices_tensor = None
  elif len(indices) == 1:
    indices_tensor = indices[0]
  else:
    if len(indices_shape) == 0:
      last_dim = 0
    else:
      last_dim = len(indices_shape) - 1
    indices_tensor = concatenate(
        indices,
        last_dim,
    )

  if indices_tensor is not None and indices_tensor.is_constant():
    lit = indices_tensor.literal_value
    # flatten all but last dim (i.e., idx/coord dim)
    coords = lit.reshape(-1, lit.shape[-1])
    unique_indices = len(np.unique(coords, axis=0)) == len(coords)
  else:
    unique_indices = advanced_indexes is None

  return _Indexer(
      newaxis_dims=tuple(newaxis_dims),
      collapsed_dims=collapsed_dims,
      indices=indices_tensor,
      unique_indices=unique_indices,
  )


def _build_gather(
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
  if indexer.indices is not None and not _is_empty(indexer.indices):
    out = gather(
        out,
        indexer.indices,
        indexer.collapsed_dims,
        unique_indices=unique_indices or indexer.unique_indices,
    )

  # This adds newaxis/None dimensions.
  return _expand_dims(out, indexer.newaxis_dims)
