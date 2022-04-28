# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file defines a dialect that mirrors the internal data structures of the
# ibis framework (https://ibis-project.org/docs/dev/). In ibis, there exists an
# additional layer of abstractions called `expressions`. Each operation in a
# query references the expressions of its subqueries and is itself reference by
# an expression. This is a remainder of the ibis UI that allows
# printing/executing/... of expressions. Notice that the additional layer of
# indirection can make ibis queries DAGs rather than trees. This only happens if
# the user enters the query in a specific way, so this is also only a left-over
# of the API. This dialect represents a cleaned up version of queries without
# the expression layer for each node. In this version, all queries are trees, so
# every part of the query is modeled as an operation with regions for its
# respective subqueries. Hence, every operation is a single region terminator
# and there are no SSAValues in this dialect.
#
# DISCLAIMER: This design tries to avoid premature complexity and is subject to
# change. In particular, comples predicates might involve operations that are
# not single region terminators.


@irdl_attr_definition
class DataType(ParametrizedAttribute):
  """
  Models an ibis datatype.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L34
  """
  name = "ibis.datatype"


@irdl_attr_definition
class Int32(DataType):
  """
  Models the ibis int32 type.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L294

  Example:

  ```
  !ibis.int32
  ```
  """
  name = "ibis.int32"


@irdl_attr_definition
class Int64(DataType):
  """
  Models the ibis int64 type.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L299

  Example:

  ```
  !ibis.int64
  ```
  """
  name = "ibis.int64"


@irdl_attr_definition
class String(DataType):
  """
  Models the ibis string type. The Parameter `nullable` defines whether the
  string can be null.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L176

  Example:

  ```
  !ibis.string<0 : !i1>
  ```
  """
  name = "ibis.string"

  nullable = ParameterDef(IntegerAttr)

  @builder
  @staticmethod
  def get(val: int) -> 'String':
    return String([IntegerAttr.from_int_and_width(val, 1)])


@irdl_op_definition
class TableColumn(Operation):
  """
  References a specific column with name `col_name` from `table`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/generic.py#L32

  Example:
  ```
  ibis.table_column() ["col_name" = "a"] {
    ibis.pandas_table() ...
  }
  ```
  """
  name = "ibis.table_column"

  table = SingleBlockRegionDef()
  col_name = AttributeDef(StringAttr)

  @builder
  @staticmethod
  def get(table: Region, col_name: str) -> 'TableColumn':
    return TableColumn.build(
        attributes={"col_name": StringAttr.from_str(col_name)}, regions=[table])


@irdl_op_definition
class Selection(Operation):
  """
  Models an SQL `Select` statement and related concepts. If there are predicates
  to filter with, they are part of `predicates`. If there is a projection, the
  wanted columns are part of `projections`. If `projections` is empty, all of
  the columns are part of the result.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/relations.py#L375

  Example:

  ```
  ibis.selection() {
    // table
    ibis.table() ...
  } {
    // predicates
    ibis.equals() ...
  } {
    // projections
  }

  ibis.selection() {
    // table
    ibis.table() ...
  } {
    // predicates
  } {
    // projections
    ibis.table_column() ...
  }
  ```
  """
  name = "ibis.selection"

  table = SingleBlockRegionDef()
  predicates = SingleBlockRegionDef()
  projections = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(table: Region, predicates: Region,
          projections: Region) -> 'Selection':
    return Selection.build(regions=[table, predicates, projections])


@irdl_op_definition
class Aggregation(Operation):
  """
  Models an ibis aggregation query where `metrics` defines the aggregation function.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/relations.py#L589

  Example:

  '''
  ibis.aggregation() {
    ibis.pandas_table() ...
  } {
    ibis.sum() {
      ...
    }
  }
  '''
  """
  name = "ibis.aggregation"

  table = SingleBlockRegionDef()
  metrics = SingleBlockRegionDef()
  # TODO: figure out what the rest of these two and model them
  # by = SingleBlockRegionDef()
  # having = SingleBlockRegionDef()
  # predicates = SingleBlockRegionDef()
  # sort_keys = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(table: Region, metrics: Region) -> 'Aggregation':
    return Aggregation.build(regions=[table, metrics])


@irdl_op_definition
class Sum(Operation):
  """
  Sums up all the elements of the column given in arg based on the encompassing
  aggregation operator.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/reductions.py#L95

  Example:

  '''
  ibis.sum() {
    ibis.table_column() ["col_name" = "id"] {
      ...
    }
  }
  '''
  """
  name = "ibis.sum"

  arg = SingleBlockRegionDef()
  # TODO: figure out what where does. Some sort of filter?
  # where = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(arg: Region) -> 'Sum':
    return Sum.build(regions=[arg])


@irdl_op_definition
class Equals(Operation):
  """
  Checks whether each entry of `left` is equal to `right`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/logical.py#L79

  Example:

  ```
  ibis.equals() {
    // left
    ibis.table_column() ...
  } {
    // right
    ibis.literal() ...
  }
  ```
  """
  name = "ibis.equals"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(left: Region, right: Region) -> 'Equals':
    return Equals.build(regions=[left, right])


@irdl_op_definition
class PandasTable(Operation):
  """
  Defines a table with name `table_name` and schema `schema`. The table is
  backed by a pandas dataframe (https://pandas.pydata.org/docs/reference/frame.html).

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/backends/pandas/client.py#L282

  Example:

  ```
  ibis.pandas_table() ["table_name" = "t"] {
    ibis.schema_element() ...
    ...
  }
  ```
  """
  name = "ibis.pandas_table"

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

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/schema.py#L35

  Example:

  ```
  ibis.schema_element() ["elt_name" = "a", "elt_type" = !ibis.int32]
  ```
  """
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
  """
  Defines a literal with value `val` and type `type`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/generic.py#L225

  Example:

  ```
  ibis.literal() ["val" = "A", "type" = ibis.string<1 : !i1>]
  ```
  """
  name = "ibis.literal"

  val = AttributeDef(AnyAttr())
  type = AttributeDef(DataType)

  @builder
  @staticmethod
  def get(val: Attribute, type: DataType) -> 'Literal':
    return Literal.build(attributes={"val": val, "type": type})


@dataclass
class Ibis:
  ctx: MLContext

  def __post_init__(self: 'Ibis'):
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(Int64)

    self.ctx.register_op(PandasTable)
    self.ctx.register_op(SchemaElement)
    self.ctx.register_op(Selection)
    self.ctx.register_op(Equals)
    self.ctx.register_op(TableColumn)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Sum)
    self.ctx.register_op(Aggregation)
