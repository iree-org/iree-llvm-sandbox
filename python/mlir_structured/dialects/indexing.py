# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
import pdb
from typing import Union, Tuple

from ._indexing_ops_gen import *
from .._mlir_libs._structuredDialects.indexing import *
from .._mlir_libs import _mlirStructuredPasses as _cextStructuredPasses

from . import tensor as tensor_dialect
from . import arith as arith_dialect
from ..ir import Type, RankedTensorType, ShapedType


class Tensor(TensorValue):

  def __str__(self):
    return super().__str__().replace("Value", "Tensor")

  @classmethod
  def empty(cls, dim_sizes: Union[list[int], tuple[int, ...]], el_type: Type):
    return cls(tensor_dialect.EmptyOp(dim_sizes, el_type).result)

  def __class_getitem__(cls, dim_sizes_el_type: Tuple[Union[list[int],
                                                            tuple[int, ...]],
                                                      Type]):
    assert (len(dim_sizes_el_type) == 2
           ), f"wrong dim_sizes_el_type: {dim_sizes_el_type}"
    dim_sizes, el_type = dim_sizes_el_type
    assert isinstance(el_type, Type), f"wrong type T args for tensor: {el_type}"
    static_sizes = []
    for s in dim_sizes:
      if isinstance(s, int):
        static_sizes.append(s)
      else:
        static_sizes.append(ShapedType.get_dynamic_size())
    return RankedTensorType.get(static_sizes, el_type)

  def __getitem__(self, dims: tuple):
    dims = list(dims)
    for i, d in enumerate(dims):
      if isinstance(d, int):
        dims[i] = arith_dialect.ConstantOp.create_index(d).result

    return tensor_dialect.ExtractOp(self, dims).result
