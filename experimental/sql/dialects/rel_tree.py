# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from ast import Param
from dataclasses import dataclass
from typing import Any, List
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file contains the relational tree dialect. This dialect represents
# relational queries in a tree form, simplifying certain optimizations.


@irdl_attr_definition
class DataType(ParametrizedAttribute):
  """
  Models a datatype in a relational query.
  """
  name = "rel_tree.datatype"


@irdl_attr_definition
class Int32(DataType):
  """
  Models a int32 type in a relational query.

  Example:

  ```
  !rel_tree.int32
  ```
  """
  name = "rel_tree.int32"


@irdl_attr_definition
class String(DataType):
  """
  Models a string type in a relational query, that can be either nullable or
  not.

  Example:

  ```
  !rel_tree.string<0: !i1>
  ```
  """
  name = "rel_tree.string"

  nullable = ParameterDef(IntegerAttr)

  @staticmethod
  @builder
  def get(val: int) -> 'String':
    return String([IntegerAttr.from_int_and_width(val, 1)])


@irdl_op_definition
class Column(Operation):
  """
  References a specific column with name `col_name`.

  Example:

  ```
  rel_tree.column() ["col_name" = "a"]
  ```
  """
  name = "rel_tree.column"

  col_name = AttributeDef(StringAttr)

  @builder
  @staticmethod
  def get(table: Region, col_name: str) -> 'Column':
    return Column.build(attributes={"col_name": StringAttr.from_str(col_name)},
                        regions=[table])


@irdl_op_definition
class Predicate(Operation):
  """
  Applies the `comparator` to `left` and `right`.

  Example:

  ```
  rel_tree.predicate() ["comparator" = "="] {
    rel_tree.column() ...
  } {
    rel_tree.literal() ...
  }
  ```
  """
  name = "rel_tree.predicate"

  comparator = AttributeDef(StringAttr)
  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(comparator: str, left: Region, right: Region) -> 'Predicate':
    return Predicate.build(
        attributes={"comparator": StringAttr.from_str(comparator)},
        regions=[left, right])


@irdl_op_definition
class Selection(Operation):
  """
  Selects all tuples from `table` that fulfill `predicates`.

  Example:

  ```
  rel_tree.selection() {
    rel_tree.pandas_table() ...
  } {
    rel_tree.predicate() ...
  }
  ```
  """
  name = "rel_tree.selection"

  table = SingleBlockRegionDef()
  predicates = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(table: Region, predicates: Region) -> 'Selection':
    return Selection.build(regions=[table, predicates])


@irdl_op_definition
class PandasTable(Operation):
  """
  Defines a table with name `table_name` and schema `schema`.

  Example:

  ```
  rel_tree.pandas_table() ["table_name" = "t"] {
    rel_tree.schema_element() ...
    ...
  }
  ```
  """
  name = "rel_tree.pandas_table"

  table_name = AttributeDef(StringAttr)
  schema = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(name: str, Schema: Region) -> 'PandasTable':
    return PandasTable.build(
        attributes={"table_name": StringAttr.from_str(name)}, regions=[Schema])


@irdl_op_definition
class SchemaElement(Operation):
  """
  Defines a schema element with name `elt_name` and type `elt_type`.

  Example:

  ```
  rel_tree.schema_element() ["elt_name" = "id", "elt_type" = !rel_tree.int32]
  ```
  """
  name = "rel_tree.schema_element"

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
  """
  Defines a literal with value `val` and type `type`.

  Example:

  ```
  rel_tree.literal() ["val" = 5 : !i64, "type" = !rel_tree.int32]
  ```
  """
  name = "rel_tree.literal"

  val = AttributeDef(AnyAttr())
  type = AttributeDef(DataType)

  @builder
  @staticmethod
  def get(val: Attribute, type: DataType) -> 'Literal':
    return Literal.build(attributes={"val": val, "type": type})


@dataclass
class RelationalTree:
  ctx: MLContext

  def __post_init__(self: 'RelationalTree'):
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Int32)

    self.ctx.register_op(PandasTable)
    self.ctx.register_op(SchemaElement)
    self.ctx.register_op(Selection)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Column)
    self.ctx.register_op(Predicate)
