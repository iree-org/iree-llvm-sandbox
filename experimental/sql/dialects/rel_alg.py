# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, List, Union
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file contains the relational algebra dialect. This dialect represents
# relational queries in a tree form, simplifying certain optimizations. Apart
# form scalar datatypes, the dialect knows two different kinds of operations:
# expressions that only have expressions in their subtrees and operators that
# can have both operators and expressions in their subtrees.


class DataType(ParametrizedAttribute):
  """
  Models a datatype in a relational query.
  """
  name = "rel_alg.datatype"


@irdl_attr_definition
class Int32(DataType):
  """
  Models a int32 type in a relational query.

  Example:

  ```
  !rel_alg.int32
  ```
  """
  name = "rel_alg.int32"


@irdl_attr_definition
class Decimal(DataType):
  """
  Models a decimal type in a relational query with precision `prec` and scale
  `scale`.

  Example:

  ```
  !rel_alg.decimal<4 : !i32, 2 : !i32>
  ```
  """
  name = "rel_alg.decimal"

  prec: ParameterDef[IntegerAttr]
  scale: ParameterDef[IntegerAttr]


@irdl_attr_definition
class Timestamp(DataType):
  """
  Models a timestamp type in a relational query.

  Example:

  ```
  !rel_alg.timestamp
  ```
  """
  name = "rel_alg.timestamp"


@irdl_attr_definition
class Int64(DataType):
  """
  Models a int64 type in a relational query.

  Example:

  ```
  !rel_alg.int64
  ```
  """
  name = "rel_alg.int64"


@irdl_attr_definition
class Float64(DataType):
  """
  Models a float64 type in a relational query.

  Example:

  ```
  !rel_alg.float64
  ```
  """
  name = "rel_alg.float64"


@irdl_attr_definition
class String(DataType):
  """
  Models a string type in a relational query, that can be either nullable or
  not.

  Example:

  ```
  !rel_alg.string
  ```
  """
  name = "rel_alg.string"


@irdl_attr_definition
class Nullable(DataType):
  """
  Models a type that is nullable.

  Example:
  ```
  !rel_alg.nullable<!rel_alg.string>
  ```
  """
  name = "rel_alg.nullable"

  type: ParameterDef[DataType]


@irdl_attr_definition
class Order(ParametrizedAttribute):
  """
  Models the order of a sort key.

  Example:

  '''
  !rel_alg.order<"a", "asc">
  '''
  """
  name = "rel_alg.order"

  col: ParameterDef[StringAttr]
  order: ParameterDef[StringAttr]


class Expression(Operation):
  ...


@irdl_op_definition
class BinOp(Expression):
  """
  Computes the binary operation `operator` of `lhs` and `rhs`.

  Example:

  '''
  rel_alg.bin_op() ["operator" = "*"] {
    rel_alg.column() ...
  } {
    rel_alg.column() ...
  }
  '''
  """

  name = "rel_alg.bin_op"

  lhs = SingleBlockRegionDef()
  rhs = SingleBlockRegionDef()
  operator = AttributeDef(StringAttr)

  @builder
  @staticmethod
  def get(lhs: Region, rhs: Region, operator: str) -> 'BinOp':
    return BinOp.create(regions=[lhs, rhs],
                        attributes={"operator": StringAttr.from_str(operator)})


@irdl_op_definition
class Literal(Expression):
  """
  Defines a literal with value `val` and type `type`.

  Example:

  ```
  rel_alg.literal() ["val" = 5 : !i64, "type" = !rel_alg.int32]
  ```
  """
  name = "rel_alg.literal"

  val = AttributeDef(AnyAttr())
  type = AttributeDef(DataType)

  @builder
  @staticmethod
  def get(val: Attribute, type: DataType) -> 'Literal':
    return Literal.create(attributes={"val": val, "type": type})


@irdl_op_definition
class Column(Expression):
  """
  References a specific column with name `col_name`.

  Example:

  ```
  rel_alg.column() ["col_name" = "a"]
  ```
  """
  name = "rel_alg.column"

  col_name = AttributeDef(StringAttr)

  @builder
  @staticmethod
  def get(col_name: str) -> 'Column':
    return Column.create(attributes={"col_name": StringAttr.from_str(col_name)})


@irdl_op_definition
class Compare(Expression):
  """
  Applies the `comparator` to `left` and `right`.

  Example:

  ```
  rel_alg.compare() ["comparator" = "="] {
    rel_alg.column() ...
  } {
    rel_alg.literal() ...
  }
  ```
  """
  name = "rel_alg.compare"

  comparator = AttributeDef(StringAttr)
  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(comparator: str, left: Region, right: Region) -> 'Compare':
    return Compare.create(
        attributes={"comparator": StringAttr.from_str(comparator)},
        regions=[left, right])


class Operator(Operation):
  ...


@irdl_op_definition
class Limit(Operator):
  """
  Limits the number of tuples in `input` to `n` .

  Example:

  ```
  rel_alg.limit() ["n" = 10 : !i64] {
    ...
  }
  ```
  """
  name = "rel_alg.limit"

  input = SingleBlockRegionDef()
  n = AttributeDef(IntegerAttr)

  @staticmethod
  @builder
  def get(table: Region, n: int) -> 'Limit':
    return Limit.create(regions=[table],
                        attributes={"n": IntegerAttr.from_int_and_width(n, 64)})


@irdl_op_definition
class OrderBy(Operator):
  """
  Orders the given input by the columns in `by`.

  Example:
  '''
  rel_alg.order_by() ["by" = [!rel_alg.order<"a", "asc>, !rel_alg.order<"b", "desc">]] {
    rel_alg.table() ...
  }
  '''
  """
  name = "rel_alg.order_by"

  input = SingleBlockRegionDef()
  by = AttributeDef(ArrayAttr)

  @builder
  @staticmethod
  def get(input: Region, by: list[str], order: list[str]) -> 'OrderBy':
    return OrderBy.build(
        regions=[input],
        attributes={
            "by":
                ArrayAttr.from_list([
                    Order([StringAttr.from_str(s),
                           StringAttr.from_str(o)]) for s, o in zip(by, order)
                ])
        })


@irdl_op_definition
class Aggregate(Operator):
  """
  Groups the table `input` by the columns in `by` by aggregating the ith element
  of `col_names` by the ith element of `functions`. If `by` is empty, this
  corresponds to the ungrouped aggregation. In the case of a `count(*)`, the
  respective element in `col_names` is `""` instead of a column name.

  Example:

  '''
  rel_alg.aggregate() ["col_names = ["b", "d"], "functions" = ["sum", "any"], "res_names" = ["a", "b"], "by" = ["c"]] {
    rel_alg.table() ...
  }
  '''
  """
  name = "rel_alg.aggregate"

  input = SingleBlockRegionDef()
  col_names = AttributeDef(ArrayOfConstraint(StringAttr))
  functions = AttributeDef(ArrayOfConstraint(StringAttr))
  res_names = AttributeDef(ArrayOfConstraint(StringAttr))
  by = AttributeDef(ArrayOfConstraint(StringAttr))

  def verify_(self) -> None:
    for f in self.functions.data:
      if not f.data in ["sum", "min", "max", "avg", "count", "count_distinct"]:
        raise Exception(f"function {f.data} is not a supported function")

  @staticmethod
  @builder
  def get(input: Region, col_names: List[str], functions: List[str],
          res_names: List[str], by: List[str]) -> 'Aggregate':
    return Aggregate.create(
        regions=[input],
        attributes={
            "col_names":
                ArrayAttr.from_list([StringAttr.from_str(n) for n in col_names]
                                   ),
            "functions":
                ArrayAttr.from_list([StringAttr.from_str(f) for f in functions]
                                   ),
            "res_names":
                ArrayAttr.from_list([StringAttr.from_str(n) for n in res_names]
                                   ),
            "by":
                ArrayAttr.from_list([StringAttr.from_str(c) for c in by])
        })


@irdl_op_definition
class Select(Operator):
  """
  Selects all tuples from `table` that fulfill `predicates`.

  Example:

  ```
  rel_alg.select() {
    rel_alg.pandas_table() ...
  } {
    rel_alg.compare() ...
  }
  ```
  """
  name = "rel_alg.select"

  input = SingleBlockRegionDef()
  predicates = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(input: Region, predicates: Region) -> 'Select':
    return Select.create(regions=[input, predicates])


@irdl_op_definition
class CartesianProduct(Operator):
  """
  Computes the Cartesian product of tables `left` and `right`.

  Example:

  '''
  rel_alg.cartesian_product() {
    rel_alg.table() ...
  } {
    rel_alg.table() ...
  }
  '''
  """
  name = "rel_alg.cartesian_product"

  left = SingleBlockRegionDef()
  right = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(left: Region, right: Region) -> 'CartesianProduct':
    return CartesianProduct.create(regions=[left, right])


@irdl_op_definition
class Project(Operator):
  """
  Projects the input table s.t. the output has as the ith column a column with
  the ith name of `names` and the ith column specified in `projectons`.

  Example:
  '''
  rel_alg.project() {
    rel_alg.unbound_table() ...
  } {
    rel_alg.column() ...
    ...
  }
  '''
  """
  name = "rel_alg.project"

  input = SingleBlockRegionDef()
  projections = SingleBlockRegionDef()
  names = AttributeDef(ArrayOfConstraint(StringAttr))

  @staticmethod
  @builder
  def get(table: Region, projections: Region, names: ArrayAttr) -> 'Project':
    return Project.create(regions=[table, projections],
                          attributes={"names": names})


@irdl_op_definition
class Table(Operator):
  """
  Defines a table with name `table_name` and schema `schema`.

  Example:

  ```
  rel_alg.table() ["table_name" = "t"] {
    rel_alg.schema_element() ...
    ...
  }
  ```
  """
  name = "rel_alg.table"

  table_name = AttributeDef(StringAttr)
  schema = SingleBlockRegionDef()

  @staticmethod
  @builder
  def get(name: str, Schema: Region) -> 'Table':
    return Table.create(attributes={"table_name": StringAttr.from_str(name)},
                        regions=[Schema])


@irdl_op_definition
class SchemaElement(Operator):
  """
  Defines a schema element with name `elt_name` and type `elt_type`.

  Example:

  ```
  rel_alg.schema_element() ["elt_name" = "id", "elt_type" = !rel_alg.int32]
  ```
  """
  name = "rel_alg.schema_element"

  elt_name = AttributeDef(StringAttr)
  elt_type = AttributeDef(DataType)

  @staticmethod
  @builder
  def get(name: str, type: DataType):
    return SchemaElement.create(attributes={
        "elt_name": StringAttr.from_str(name),
        "elt_type": type
    })


@dataclass
class RelationalAlg:
  ctx: MLContext

  def __post_init__(self: 'RelationalAlg'):
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(Timestamp)
    self.ctx.register_attr(Decimal)
    self.ctx.register_attr(Int64)
    self.ctx.register_attr(Nullable)
    self.ctx.register_attr(Order)
    self.ctx.register_attr(Float64)

    self.ctx.register_op(Table)
    self.ctx.register_op(Limit)
    self.ctx.register_op(SchemaElement)
    self.ctx.register_op(Select)
    self.ctx.register_op(Project)
    self.ctx.register_op(CartesianProduct)
    self.ctx.register_op(BinOp)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Column)
    self.ctx.register_op(Compare)
    self.ctx.register_op(Aggregate)
    self.ctx.register_op(OrderBy)
