# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file define a dialect that mirrors the internal structure of ibis.
# The file contains an `Attribute` for all types encountered in the simple
# query. Types are defined as being in defined `ibis/expr/types` in the
# ibis source repo. All other nodes encountered in this translation are
# encoded as Operations.


@irdl_attr_definition
class TableExpr(ParametrizedAttribute):
  name = "ibis.table_expr"


@irdl_attr_definition
class Column(ParametrizedAttribute):
  name = "ibis.column"


@irdl_attr_definition
class StringColumn(Column):
  name = "ibis.string_column"


@irdl_attr_definition
class BooleanColumn(Column):
  name = "ibis.boolean_column"


@irdl_attr_definition
class DataType(ParametrizedAttribute):
  name = "ibis.datatype"


@irdl_attr_definition
class int32(DataType):
  name = "ibis.int32"


@irdl_attr_definition
class String(DataType):
  name = "ibis.String"

  nullable = ParameterDef(IntegerAttr)

  @builder
  @staticmethod
  def get(val: int) -> 'String':
    return String([IntegerAttr.from_int_and_width(val, 1)])


@irdl_attr_definition
class float64(DataType):
  name = "ibis.float64"


@irdl_op_definition
class TableColumn(Operation):
  name = "ibis.table_column"

  table = SingleBlockRegionDef()
  col_name = AttributeDef(StringAttr)

  result = ResultDef(AnyAttr())

  @builder
  @staticmethod
  def get(table: Region, col_name: str,
          result_type: Attribute) -> 'TableColumn':
    return TableColumn.build(
        attributes={"col_name": StringAttr.from_str(col_name)},
        regions=[table],
        result_types=[result_type])


@irdl_op_definition
class Selection(Operation):
  name = "ibis.selection"

  table = SingleBlockRegionDef()
  predicates = SingleBlockRegionDef()

  result = ResultDef(TableExpr)

  @staticmethod
  @builder
  def get(table: Region, predicates: Region) -> 'Selection':
    return Selection.build(regions=[table, predicates],
                           result_types=[TableExpr()])


@irdl_op_definition
class Equals(Operation):
  name = "ibis.equals"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  result = ResultDef(BooleanColumn)

  @builder
  @staticmethod
  def get(left: Region, right: Region) -> 'Equals':
    return Equals.build(regions=[left, right], result_types=[BooleanColumn()])


@irdl_op_definition
class PandasTable(Operation):
  name = "ibis.pandas_table"

  table_name = AttributeDef(StringAttr)
  schema = SingleBlockRegionDef()
  result = ResultDef(TableExpr)

  @staticmethod
  @builder
  def get(name: str, Schema: Region) -> 'PandasTable':
    return PandasTable.build(
        attributes={"table_name": StringAttr.from_str(name)},
        regions=[Schema],
        result_types=[TableExpr()])


@irdl_op_definition
class SchemaElement(Operation):
  name = "ibis.schema_element"

  elt_name = AttributeDef(StringAttr)
  elt_type = AttributeDef(DataType)

  @staticmethod
  def get(name: str, type: DataType):
    return SchemaElement.build(attributes={
        "elt_name": StringAttr.from_str(name),
        "elt_type": type
    })


@irdl_op_definition
class Literal(Operation):
  name = "ibis.literal"

  val = AttributeDef(AnyAttr())
  type = AttributeDef(DataType)

  result = ResultDef(DataType)

  @builder
  @staticmethod
  def get(val: Attribute, type: DataType) -> 'Literal':
    return Literal.build(attributes={
        "val": val,
        "type": type
    },
                         result_types=[type])


@irdl_op_definition
class Yield(Operation):
  name = "ibis.yield"

  returns = VarOperandDef(AnyAttr())

  @staticmethod
  @builder
  def get(ops: list[Operation]) -> 'Yield':
    return Yield.build(operands=[ops])


@dataclass
class Ibis:
  ctx: MLContext

  def __post_init__(self: 'Ibis'):
    self.ctx.register_attr(TableExpr)
    self.ctx.register_attr(Column)
    self.ctx.register_attr(StringColumn)
    self.ctx.register_attr(BooleanColumn)
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(String)
    self.ctx.register_attr(int32)
    self.ctx.register_attr(float64)

    self.ctx.register_op(PandasTable)
    self.ctx.register_op(SchemaElement)
    self.ctx.register_op(Selection)
    self.ctx.register_op(Equals)
    self.ctx.register_op(TableColumn)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Yield)
