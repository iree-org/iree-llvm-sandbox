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


class DataType(ParametrizedAttribute):
  """
  Models an ibis datatype.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L34
  """
  name = "ibis.datatype"


@irdl_attr_definition
class Nullable(DataType):
  """
  Models the fact that an ibis datatype is nullable.
  """
  name = "ibis.nullable"

  datatype: ParameterDef[DataType]


@irdl_attr_definition
class Decimal(DataType):
  """
  Models the ibis decimal type.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L344
  """
  name = "ibis.decimal"

  prec: ParameterDef[IntegerAttr]
  scale: ParameterDef[IntegerAttr]


@irdl_attr_definition
class Timestamp(DataType):
  """
  Models the ibis timestamp type.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L222
  """
  name = "ibis.timestamp"


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
class Float64(DataType):
  """
  Models the ibis float64 type.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L334

  Example:

  ```
  !ibis.float64
  ```
  """
  name = "ibis.float64"


@irdl_attr_definition
class String(DataType):
  """
  Models the ibis string type. The Parameter `nullable` defines whether the
  string can be null.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/datatypes.py#L176

  Example:

  ```
  !ibis.string
  ```
  """
  name = "ibis.string"


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
    return TableColumn.create(
        attributes={"col_name": StringAttr.from_str(col_name)}, regions=[table])


@irdl_op_definition
class Subtract(Operation):
  """
  Models a subtraction of the columns `lhs` and `rhs` with result type `output_type`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/numeric.py#L39

  Example:
  ```
  ibis.subtraction() ["output_type" = !ibis.int64] {
    // lhs
    ibis.table_column() ...
  } {
    // rhs
    ibis.table_column() ...
  }
  ```

  """
  name = "ibis.subtract"

  lhs = SingleBlockRegionDef()
  rhs = SingleBlockRegionDef()
  output_type = AttributeDef(DataType)

  @staticmethod
  @builder
  def get(lhs: Region, rhs: Region, output_type: DataType) -> 'Subtract':
    return Subtract.create(regions=[lhs, rhs],
                           attributes={"output_type": output_type})


@irdl_op_definition
class Add(Operation):
  """
  Models an addition of the columns `lhs` and `rhs` with result type `output_type`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/numeric.py#L20

  Example:
  ```
  ibis.add() ["output_type" = !ibis.int64] {
    // lhs
    ibis.table_column() ...
  } {
    // rhs
    ibis.table_column() ...
  }
  ```

  """
  name = "ibis.add"

  lhs = SingleBlockRegionDef()
  rhs = SingleBlockRegionDef()
  output_type = AttributeDef(DataType)

  @staticmethod
  @builder
  def get(lhs: Region, rhs: Region, output_type: DataType) -> 'Add':
    return Add.create(regions=[lhs, rhs],
                      attributes={"output_type": output_type})


@irdl_op_definition
class Divide(Operation):
  """
  Models a division of the columns `lhs` and `rhs` with result type `output_type`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/numeric.py#L44

  Example:
  ```
  ibis.divide() ["output_type" = !ibis.int64] {
    // lhs
    ibis.table_column() ...
  } {
    // rhs
    ibis.table_column() ...
  }
  ```

  """
  name = "ibis.divide"

  lhs = SingleBlockRegionDef()
  rhs = SingleBlockRegionDef()
  output_type = AttributeDef(DataType)

  @staticmethod
  @builder
  def get(lhs: Region, rhs: Region, output_type: DataType) -> 'Divide':
    return Divide.create(regions=[lhs, rhs],
                         attributes={"output_type": output_type})


@irdl_op_definition
class Multiply(Operation):
  """
  Models an multiplication of the columnds `lhs` and `rhs` with result type `output_type`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/numeric.py#L25

  Example:
  ```
  ibis.selecton() ["names" = ["res]] {
    ibis.unbound_table() ...
  } {} {
    ibis.multiply() ["output_type" = !ibis.int64] {
      // lhs
      ibis.table_column() ...
    } {
      // rhs
      ibis.table_column() ...
    }
  }
  ```

  """
  name = "ibis.multiply"

  lhs = SingleBlockRegionDef()
  rhs = SingleBlockRegionDef()
  output_type = AttributeDef(DataType)

  @staticmethod
  @builder
  def get(lhs: Region, rhs: Region, output_type: DataType) -> 'Multiply':
    return Multiply.create(regions=[lhs, rhs],
                           attributes={"output_type": output_type})


@irdl_op_definition
class SortKey(Operation):
  """
  Models a sort key in ibis. The main information in this node is the order
  ("asc" for ascending and "desc" for desceding).

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/sortkeys.py#L60

  Example:

  '''
  ibis.sort_key() ["order" = "asc"] {
    ibis.table_column() ["col_name" = "a"] ...
  }
  '''
  """
  name = "ibis.sort_key"

  expr = SingleBlockRegionDef()
  order = AttributeDef(AnyAttr())

  @staticmethod
  @builder
  def get(expr: Region, asc: bool) -> 'SortKey':
    return SortKey.create(regions=[expr],
                          attributes={
                              "order":
                                  StringAttr.from_str("asc")
                                  if asc else StringAttr.from_str("desc")
                          })


@irdl_op_definition
class Selection(Operation):
  """
  Models an SQL `Select` statement and related concepts. If there are predicates
  to filter with, they are part of `predicates`. If there is a projection, the
  wanted columns are part of `projections`. The ith projected column will have
  the ith name of `names`. If `projections` is empty, all of the columns are
  part of the result.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/relations.py#L375

  Example:

  ```
  ibis.selection() ["names" = []] {
    // table
    ibis.table() ...
  } {
    // predicates
    ibis.equals() ...
  } {
    // projections
  }

  ibis.selection() ["names" = ["c"]] {
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
  sort_keys = SingleBlockRegionDef()
  names = AttributeDef(ArrayOfConstraint(StringAttr))

  @staticmethod
  @builder
  def get(table: Region, predicates: Region, projections: Region,
          sort_keys: Region, names: list[str]) -> 'Selection':
    return Selection.create(regions=[table, predicates, projections, sort_keys],
                            attributes={
                                "names":
                                    ArrayAttr.from_list(
                                        [StringAttr.from_str(n) for n in names])
                            })


@irdl_op_definition
class Aggregation(Operation):
  """
  Models an ibis aggregation over `table` where `metrics` defines the
  aggregation function, `names` defines the result schema, and `by` defines the
  columns to group by. If `by` is empty, this corresponds to a ungrouped
  aggregation. If there is a `TableColumn` as hte top-level operation of
  `metrics`, we use an implicit `ANY` aggregation, so the result will just
  correspond to any value of that column. If `metrics` is empty, this
  corresponds to choosing all columns. The ones not grouped by are aggregate
  through `ANY`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/relations.py#L589

  Example:

  '''
  ibis.aggregation() ["names" = ["a", "b"]] {
    ibis.unbound_table() ...
  } {
    ibis.sum() {
      ...
    }
  } {
    ibis.table_column() ...
  }
  '''

  TODO: add support for grouping without aggregation
  """
  name = "ibis.aggregation"

  table = SingleBlockRegionDef()
  metrics = SingleBlockRegionDef()
  names = AttributeDef(ArrayOfConstraint(StringAttr))
  by = SingleBlockRegionDef()
  # TODO: figure out what the rest of these two and model them
  # having = SingleBlockRegionDef()
  # predicates = SingleBlockRegionDef()
  # sort_keys = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(table: Region, metrics: Region, by: Region,
          names: list[str]) -> 'Aggregation':
    return Aggregation.create(
        regions=[table, metrics, by],
        attributes={
            "names":
                ArrayAttr.from_list([StringAttr.from_str(n) for n in names])
        })


@irdl_op_definition
class CartesianProduct(Operation):
  """
  Models an ibis Cartesian product of tables `left` and `right`, i.e. the
  combination of all elements for `left` and `right`.

  Example:

  '''
  ibis.cartesian_product() {
    ibis.unbound_table() ...
  } {
    ibis.unbound_table() ...
  }
  '''
  """
  name = "ibis.cartesian_product"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(left: Region, right: Region) -> 'CartesianProduct':
    return CartesianProduct.create(regions=[left, right])


@irdl_op_definition
class Sum(Operation):
  """
  Sums up all the elements of the column given in arg based on the encompassing
  aggregation operator. The operation in `arg` has to be a TableColumn.

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
    return Sum.create(regions=[arg])


@irdl_op_definition
class Min(Operation):
  """
  Takes the minimum of all the elements of the column given in `arg` based on
  the encompassing aggregation operator. The operation in `arg` has to be a
  TableColumn.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/reductions.py#L204

  Example:

  '''
  ibis.min() {
    ibis.table_column() ...
  }
  '''
  """
  name = "ibis.min"

  arg = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(arg: Region) -> 'Min':
    return Min.create(regions=[arg])


@irdl_op_definition
class Max(Operation):
  """
  Takes the maximum of all the elements of the column given in arg based on the
  encompassing aggregation operator. The operation in `arg` has to be a
  TableColumn.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/reductions.py#L197

  Example:

  '''
  ibis.max() {
    ibis.table_column() ...
  }
  '''
  """
  name = "ibis.max"

  arg = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(arg: Region) -> 'Max':
    return Max.create(regions=[arg])


@irdl_op_definition
class Mean(Operation):
  """
  Takes the mean of all the elements of the column given in arg based on the
  encompassing aggregation operator. The operation in `arg` has to be a
  TableColumn.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/reductions.py#L108

  Example:

  '''
  ibis.mean() {
    ibis.table_column() ...
  }
  '''
  """
  name = "ibis.mean"

  arg = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(arg: Region) -> 'Mean':
    return Mean.create(regions=[arg])


@irdl_op_definition
class Count(Operation):
  """
  Counts the elements in arg based on the encompassing aggregation operator. The
  operation in `arg` has to be either a TableColumn for the count of non null
  values in that column or a Table in the case of count(*).

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/reductions.py#L18

  Example:

  '''
  ibis.count() {
    ibis.unbound_table() ...
  }
  '''
  """
  name = "ibis.count"

  arg = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(arg: Region) -> 'Count':
    return Count.create(regions=[arg])


@irdl_op_definition
class CountDistinct(Operation):
  """
  Counts the distinct elements of the column in arg based on the encompassing
  aggregation operator. The operation in `arg` has to be a TableColumn.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/reductions.py#L265

  Example:

  '''
  ibis.count_distinct() {
    ibis.table_column() ...
  }
  '''
  """
  name = "ibis.count_distinct"

  arg = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(arg: Region) -> 'CountDistinct':
    return CountDistinct.create(regions=[arg])


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
    return Equals.create(regions=[left, right])


@irdl_op_definition
class GreaterEqual(Operation):
  """
  Checks whether each entry of `left` is greater or equal to `right`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/logical.py#L89

  Example:

  ```
  ibis.greaterEqual() {
    // left
    ibis.table_column() ...
  } {
    // right
    ibis.literal() ...
  }
  ```
  """
  name = "ibis.greaterEqual"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(left: Region, right: Region) -> 'GreaterEqual':
    return GreaterEqual.create(regions=[left, right])


@irdl_op_definition
class GreaterThan(Operation):
  """
  Checks whether each entry of `left` is greater than `right`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/logical.py#L94

  Example:

  ```
  ibis.greaterThan() {
    // left
    ibis.table_column() ...
  } {
    // right
    ibis.literal() ...
  }
  ```
  """
  name = "ibis.greaterThan"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(left: Region, right: Region) -> 'GreaterThan':
    return GreaterThan.create(regions=[left, right])


@irdl_op_definition
class LessThan(Operation):
  """
  Checks whether each entry of `left` is less than `right`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/logical.py#L104

  Example:

  ```
  ibis.lessThan() {
    // left
    ibis.table_column() ...
  } {
    // right
    ibis.literal() ...
  }
  ```
  """
  name = "ibis.lessThan"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(left: Region, right: Region) -> 'LessThan':
    return LessThan.create(regions=[left, right])


@irdl_op_definition
class LessEqual(Operation):
  """
  Checks whether each entry of `left` is less or equal to `right`.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/logical.py#L99

  Example:

  ```
  ibis.lessEqual() {
    // left
    ibis.table_column() ...
  } {
    // right
    ibis.literal() ...
  }
  ```
  """
  name = "ibis.lessEqual"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(left: Region, right: Region) -> 'LessEqual':
    return LessEqual.create(regions=[left, right])


@irdl_op_definition
class UnboundTable(Operation):
  """
  Defines a table with name `table_name` and schema `schema`. The table is
  unbound, i.e. it is not backed by data.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/relations.py#L70

  Example:

  ``` ibis.unbound_table() ["table_name" = "t"] {
    ibis.schema_element() ...
    ...
  }
  ```
  """
  name = "ibis.unbound_table"

  table_name = AttributeDef(StringAttr)
  schema = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(name: str, Schema: Region) -> 'UnboundTable':
    return UnboundTable.create(
        attributes={"table_name": StringAttr.from_str(name)}, regions=[Schema])


@irdl_op_definition
class Limit(Operation):
  """
  Limits the number of tuples in `table` to `n` .

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/relations.py#L337

  Example:

  ```
  ibis.limit() ["n" = 10 : !i64] {
    ...
  }
  ```
  """
  name = "ibis.limit"

  table = SingleBlockRegionDef()
  n = AttributeDef(IntegerAttr)

  @staticmethod
  @builder
  def get(table: Region, n: int) -> 'Limit':
    return Limit.create(regions=[table],
                        attributes={"n": IntegerAttr.from_int_and_width(n, 64)})


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

  @builder
  @staticmethod
  def get(name: str, type: DataType):
    return SchemaElement.create(attributes={
        "elt_name": StringAttr.from_str(name),
        "elt_type": type
    })


@irdl_op_definition
class Between(Operation):
  """
  Returns whether `arg` is in between `lower_bound` and `upper_bound`. These bounds are inclusive.

  https://github.com/ibis-project/ibis/blob/f3d267b96b9f14d3616c17b8f7bdeb8d0a6fc2cf/ibis/expr/operations/logical.py#L114

  Example:
  '''
  ibis.between() {
    ibis.table_colum()...
  } {
    ibis.literal() ...
  } {
    ibis.literal()
  }
  '''
  """
  name = "ibis.between"

  arg = SingleBlockRegionDef()
  lower_bound = SingleBlockRegionDef()
  upper_bound = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(arg: Region, lower_bound: Region, upper_bound: Region) -> 'Between':
    return Between.create(regions=[arg, lower_bound, upper_bound])


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
    return Literal.create(attributes={"val": val, "type": type})


@dataclass
class Ibis:
  ctx: MLContext

  def __post_init__(self: 'Ibis'):
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(Int64)
    self.ctx.register_attr(Float64)
    self.ctx.register_attr(Decimal)
    self.ctx.register_attr(Timestamp)
    self.ctx.register_attr(Nullable)

    self.ctx.register_op(Subtract)
    self.ctx.register_op(Add)
    self.ctx.register_op(Divide)
    self.ctx.register_op(UnboundTable)
    self.ctx.register_op(SortKey)
    self.ctx.register_op(Limit)
    self.ctx.register_op(SchemaElement)
    self.ctx.register_op(Selection)
    self.ctx.register_op(CartesianProduct)
    self.ctx.register_op(Multiply)
    self.ctx.register_op(Between)
    self.ctx.register_op(Equals)
    self.ctx.register_op(GreaterEqual)
    self.ctx.register_op(GreaterThan)
    self.ctx.register_op(LessEqual)
    self.ctx.register_op(LessThan)
    self.ctx.register_op(TableColumn)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Sum)
    self.ctx.register_op(Mean)
    self.ctx.register_op(Min)
    self.ctx.register_op(Max)
    self.ctx.register_op(Count)
    self.ctx.register_op(CountDistinct)
    self.ctx.register_op(Aggregation)
