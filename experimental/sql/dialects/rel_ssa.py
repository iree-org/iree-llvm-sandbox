# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Union, List
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file contains the relational dialect. It is currently
# only aimed at the simple query and mirrors a lot of the
# ibis dialect. This is subject to change, once the translation
# to the iterator dialect starts getting in shape.


@irdl_attr_definition
class DataType(ParametrizedAttribute):
  name = "rel_ssa.datatype"


@irdl_attr_definition
class Int32(DataType):
  name = "rel_ssa.int32"


@irdl_attr_definition
class String(DataType):
  name = "rel_ssa.string"

  nullable = ParameterDef(IntegerAttr)

  @staticmethod
  @builder
  def get(val: Union[int, IntegerAttr]) -> 'String':
    if isinstance(val, IntegerAttr):
      return String([val])
    return String([IntegerAttr.from_int_and_width(val, 1)])


@irdl_attr_definition
class Boolean(ParametrizedAttribute):
  name = "rel_ssa.bool"


@irdl_attr_definition
class Bag(ParametrizedAttribute):
  name = "rel_ssa.bag"

  schema = ParameterDef(ArrayOfConstraint(DataType))

  @staticmethod
  @builder
  def get(types: list[DataType]) -> 'Bag':
    return Bag([ArrayAttr.from_list(types)])  #type: ignore


@irdl_op_definition
class Column(Operation):
  name = "rel_ssa.column"

  col_name = AttributeDef(StringAttr)
  result = ResultDef(DataType)


@irdl_op_definition
class Compare(Operation):
  name = "rel_ssa.compare"

  left = OperandDef(AnyAttr())
  right = OperandDef(AnyAttr())
  comparator = AttributeDef(StringAttr)

  result = ResultDef(AnyAttr())

  @staticmethod
  @builder
  def get(left: Operation, right: Operation,
          comparator: StringAttr) -> 'Compare':
    return Compare.build(operands=[left, right],
                         attributes={"comparator": comparator},
                         result_types=[AnyAttr()])


@irdl_op_definition
class PandasTable(Operation):
  name = "rel_ssa.pandas_table"

  table_name = AttributeDef(StringAttr)
  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(name: str, result_type: Attribute) -> 'PandasTable':
    return PandasTable.build(
        attributes={"table_name": StringAttr.from_str(name)},
        result_types=[result_type])


@irdl_op_definition
class Select(Operation):
  name = "rel_ssa.select"

  input = OperandDef(Bag)
  predicates = SingleBlockRegionDef()

  result = ResultDef(Bag)

  @builder
  @staticmethod
  def get(table: Operation, predicates: Region) -> 'Select':
    return Select.build(operands=[table],
                        regions=[predicates],
                        result_types=[Bag()])


@irdl_op_definition
class Yield(Operation):
  name = "rel_ssa.yield"

  ops = VarOperandDef(AnyAttr())

  @staticmethod
  @builder
  def get(ops: list[Operation]) -> 'Yield':
    return Yield.build(operands=[ops])


@irdl_op_definition
class Literal(Operation):
  name = "rel_ssa.literal"

  # TODO: change IntegerAttr s.t. it can have type !rel.int32
  value = AttributeDef(AnyAttr())
  result = ResultDef(DataType)

  @staticmethod
  @builder
  def get(val: Attribute, res: DataType) -> 'Literal':
    return Literal.build(attributes={"value": val}, result_types=[res])


@dataclass
class RelSSA:
  ctx: MLContext

  def __post_init__(self: 'RelSSA'):
    self.ctx.register_attr(Bag)
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Boolean)

    self.ctx.register_op(Select)
    self.ctx.register_op(PandasTable)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Compare)
    self.ctx.register_op(Column)
    self.ctx.register_op(Yield)
