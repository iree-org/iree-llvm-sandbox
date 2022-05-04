# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Union, List
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr, IntegerType, TupleType
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

#===------------------------------------------------------------------------===#
# Data types
#===------------------------------------------------------------------------===#

#===------------------------------------------------------------------------===#
# Query types
#===------------------------------------------------------------------------===#


@irdl_attr_definition
class Stream(ParametrizedAttribute):
  """
  Stream of elements of the given type.
  """
  name = "iterators.stream"

  types = ParameterDef(TupleType)

  @builder
  @staticmethod
  def get(elem_type: List[TupleType]) -> 'Stream':
    return Stream([elem_type])  #type: ignore


#===------------------------------------------------------------------------===#
# Operations
#===------------------------------------------------------------------------===#


@irdl_op_definition
class SampleInputOp(Operation):
  """
  Create some sample input.
  """
  name = "iterators.sampleInput"

  result = ResultDef(Stream)

  @builder
  @staticmethod
  def get(input: Operation) -> 'SampleInputOp':
    return SampleInputOp.build(operands=[input])


@irdl_op_definition
class ReduceOp(Operation):
  """
  Reduce the input to a single tuple.
  """
  name = "iterators.reduce"

  arguments = OperandDef(Stream)

  result = ResultDef(Stream)

  @builder
  @staticmethod
  def get(argument: Operation) -> 'ReduceOp':
    return ReduceOp.build(operands=[argument],
                          result_types=[Stream([IntegerType.from_width(32)])])


@irdl_op_definition
class SinkOp(Operation):
  """
  Consume tuples from an iterator.
  """
  name = "iterators.sink"

  arguments = OperandDef(Stream)

  @builder
  @staticmethod
  def get(arguments: Operation) -> 'SinkOp':
    return SinkOp.build(operands=[arguments])


@dataclass
class Iterators:
  ctx: MLContext

  def __post_init__(self: 'Iterators'):
    self.ctx.register_attr(Stream)

    self.ctx.register_op(SampleInputOp)
    self.ctx.register_op(ReduceOp)
    self.ctx.register_op(SinkOp)
