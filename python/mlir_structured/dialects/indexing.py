# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import operator
import re
from copy import deepcopy
from functools import cached_property, lru_cache, partialmethod
from typing import Union, Tuple, Optional

import numpy as np

from . import arith, tensor
from ._arith_ops_ext import _is_integer_like_type
from ._indexing_ops_gen import *
from ._ods_common import get_op_result_or_value
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

  def __getitem__(self, dims: tuple) -> Scalar:
    dims = list(dims)
    for i, d in enumerate(dims):
      if isinstance(d, int):
        dims[i] = arith.ConstantOp.create_index(d).result

    return Scalar(tensor.ExtractOp(self, dims))
