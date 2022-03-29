# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file contains the relational dialect. It is currently
# only aimed at the simple query and mirrors a lot of the
# ibis dialect. This is subject to change, once the translation
# to the iterator dialect starts getting in shape.


@irdl_attr_definition
class DataType(ParametrizedAttribute):
  name = "rel.datatype"


@irdl_attr_definition
class Int32(DataType):
  name = "rel.int32"


@irdl_attr_definition
class String(DataType):
  name = "rel.string"

  nullable = ParameterDef(IntegerAttr)

  @staticmethod
  @builder
  def get(val: int) -> 'String':
    return String([IntegerAttr.from_int_and_width(val, 1)])


@irdl_attr_definition
class Float64(DataType):
  name = "rel.float64"


@irdl_attr_definition
class Bag(ParametrizedAttribute):
  name = "rel.bag"


@irdl_attr_definition
class Column(ParametrizedAttribute):
  name = "rel.column"


@irdl_op_definition
class Equals(Operation):
  name = "rel.equals"

  table = OperandDef(Bag)
  column = AttributeDef(StringAttr)
  comparator = OperandDef(AnyAttr())

  result = ResultDef(Column())

  @staticmethod
  @builder
  def get(table: Operation, column: StringAttr,
          comparator: Operation) -> 'Equals':
    return Equals.build(operands=[table, comparator],
                        attributes={"column": column},
                        result_types=[Column()])


@irdl_op_definition
class PandasTable(Operation):
  name = "rel.pandas_table"

  table_name = AttributeDef(StringAttr)
  schema = SingleBlockRegionDef()
  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(name: str, Schema: Region, result_type: Attribute) -> 'PandasTable':
    return PandasTable.build(
        attributes={"table_name": StringAttr.from_str(name)},
        regions=[Schema],
        result_types=[result_type])


@irdl_op_definition
class SchemaElement(Operation):
  name = "rel.schema_element"

  elt_name = AttributeDef(StringAttr)
  elt_type = AttributeDef(DataType)

  @staticmethod
  def get(name: str, type: DataType):
    return SchemaElement.build(attributes={
        "elt_name": StringAttr.from_str(name),
        "elt_type": type
    })


@irdl_op_definition
class Selection(Operation):
  name = "rel.selection"

  parent_ = OperandDef(Bag)
  predicates = SingleBlockRegionDef()

  result = ResultDef(Bag)

  @builder
  @staticmethod
  def get(table: Operation, predicates: Region) -> 'Selection':
    return Selection.build(operands=[table],
                           regions=[predicates],
                           result_types=[Bag()])


@irdl_op_definition
class Yield(Operation):
  name = "rel.yield"

  ops = VarOperandDef(AnyAttr())

  @staticmethod
  @builder
  def get(ops: list[Operation]) -> 'Yield':
    return Yield.build(operands=[ops])


@irdl_op_definition
class Literal(Operation):
  name = "rel.literal"

  # TODO: change IntegerAttr s.t. it can have type !rel.int32
  value = AttributeDef(AnyAttr())
  result = ResultDef(DataType)

  @staticmethod
  @builder
  def get(val: Attribute, res: DataType) -> 'Literal':
    return Literal.build(attributes={"value": val}, result_types=[res])


@dataclass
class Relational:
  ctx: MLContext

  def __post_init__(self: 'Relational'):
    self.ctx.register_attr(Bag)
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(Float64)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Column)

    self.ctx.register_op(Selection)
    self.ctx.register_op(PandasTable)
    self.ctx.register_op(SchemaElement)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Equals)
