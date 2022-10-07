# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Union, List, Optional
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder, OptAttributeDef

# This file contains the relational implementation dialect, a dialect that
# expresses relational queries using low-level concepts like tuples as block
# arguments and specific join implementations. The dialect consists of two types
# of operations: Expressions and Operators. Operators work on Bags whereas
# expressions work on scalars and are used to specify the exact behavior of
# operators. The IndexByName and Yield Expressions bridge the gap to convert
# from one to the other.

#===------------------------------------------------------------------------===#
# Data types
#===------------------------------------------------------------------------===#


class DataType(ParametrizedAttribute):
  """
  Models a datatype in a relational implementation query.
  """
  name = "rel_impl.datatype"


@irdl_attr_definition
class Int32(DataType):
  """
  Models a int32 type in a relational implementation query.

  Example:

  ```
  !rel_impl.int32
  ```
  """
  name = "rel_impl.int32"


@irdl_attr_definition
class Int64(DataType):
  """
  Models a int64 type in a relational implementation query.

  Example:

  ```
  !rel_impl.int64
  ```
  """
  name = "rel_impl.int64"


@irdl_attr_definition
class Float64(DataType):
  """
  Models a float64 type in a relational implementation query.

  Example:

  ```
  !rel_impl.float64
  ```
  """
  name = "rel_impl.float64"


@irdl_attr_definition
class Decimal(DataType):
  """
  Models a decimal type in a relational implementation query with precision `prec` and scale `scale`.

  Example:

  ```
  !rel_impl.decimal<4 : !i32, 2 : !i32>
  ```
  """
  name = "rel_impl.decimal"

  prec: ParameterDef[IntegerAttr]
  scale: ParameterDef[IntegerAttr]


@irdl_attr_definition
class Timestamp(DataType):
  """
  Models a timestamp type in a relational implementation query.

  Example:

  ```
  !rel_impl.timestamp
  ```
  """
  name = "rel_impl.timestamp"


@irdl_attr_definition
class String(DataType):
  """
  Models a string type in a relational implementation query, that can be either
  nullable or not.

  Example:

  ```
  !rel_impl.string
  ```
  """
  name = "rel_impl.string"


@irdl_attr_definition
class Nullable(DataType):
  """
  Models a type that is nullable.

  Example:
  ```
  !rel_impl.nullable<!rel_impl.string>
  ```
  """
  name = "rel_impl.nullable"

  type: ParameterDef[DataType]


#===------------------------------------------------------------------------===#
# Query types
#===------------------------------------------------------------------------===#


@irdl_attr_definition
class Order(ParametrizedAttribute):
  """
  Models the order of a sort key.

  Example:

  '''
  !rel_impl.order<"a", "asc">
  '''
  """
  name = "rel_impl.order"

  col: ParameterDef[StringAttr]
  order: ParameterDef[StringAttr]


@irdl_attr_definition
class Boolean(ParametrizedAttribute):
  """
  Models a type that can either be true or false to, e.g., show whether a tuple
  is part of the result or not.

  Example:

  '''
  !rel_impl.bool
  '''
  """
  name = "rel_impl.bool"


@irdl_attr_definition
class SchemaElement(ParametrizedAttribute):
  """
  Models an element of a schema with name `elt_name` and type `elt_type`.

  Example:
  '''
  !rel_impl.schema_element<"id", !rel_impl.int32>
  '''
  """
  name = "rel_impl.schema_element"

  elt_name: ParameterDef[StringAttr]
  elt_type: ParameterDef[DataType]

  @staticmethod
  @builder
  def get(name: str, type_: DataType) -> 'SchemaElement':
    return SchemaElement([StringAttr.from_str(name), type_])


@irdl_attr_definition
class Bag(ParametrizedAttribute):
  """
  Models a bag in a relational implementation query. The exact schema of the bag
  is part of the type itself.

  Example:

  '''
  !rel_impl.bag<[!rel_impl.schema_element<"id", !rel_impl.int32>]>
  '''
  """
  name = "rel_impl.bag"

  schema: ParameterDef[ArrayAttr[SchemaElement]]

  @staticmethod
  @builder
  def get(types: list[DataType], names: list[str]) -> 'Bag':
    schema_elts = [SchemaElement.get(n, t) for n, t in zip(names, types)]
    return Bag([ArrayAttr.from_list(schema_elts)])

  def lookup_type_in_schema(self, name: str) -> Optional[DataType]:
    """
    Looks up the type of name in the schema of this bag.
    """
    for s in self.schema.data:
      if s.elt_name.data == name:
        return s.elt_type
    return None


@irdl_attr_definition
class Tuple(ParametrizedAttribute):
  """
  Models a tuple in a relational implementation query. The schema is part of the
  type of the tuple.

  Example:

  '''
  !rel_impl.tuple<[!rel_impl.schema_element<"id", !rel_impl.int32>]>
  '''
  """
  name = "rel_impl.tuple"

  schema: ParameterDef[ArrayAttr[SchemaElement]]

  @staticmethod
  @builder
  def get(types: list[DataType], names: list[str]) -> 'Tuple':
    schema_elts = [SchemaElement.get(n, t) for n, t in zip(names, types)]
    return Tuple([ArrayAttr.from_list(schema_elts)])


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#


class Expression(Operation):
  """
  Interface class for all Expressions, i.e., operations that work on single
  values.
  """
  ...


@irdl_op_definition
class IndexByName(Expression):
  """
  Indexes into `tuple` at the position of the field `col_name` and returns its
  content.

  Example:

  '''
   %0 : rel_impl.int32 = rel_impl.index_by_name(%t : !rel_impl.tuple<[!rel_impl.schema_element<"id", !rel_impl.int32>]>) ["col_name" = "id"]
    '''
  """
  name = "rel_impl.index_by_name"

  col_name = AttributeDef(StringAttr)
  tuple = OperandDef(Tuple)
  result = ResultDef(DataType)

  @builder
  @staticmethod
  def get(name: str, tuple: Tuple, res_type: DataType) -> 'IndexByName':
    return IndexByName.create(
        operands=[tuple],
        result_types=[res_type],
        attributes={"col_name": StringAttr.from_str(name)})


@irdl_op_definition
class BinOp(Expression):
  """
  Models a binary operation of `lhs` `operator` `rhs`.

  Example:

  '''
  %1 : !rel_impl.int32 = rel_impl.bin_op(%0 : !rel_impl.int32, %2 : !rel_impl.int32) ["operator" = "*"]
  '''
  """
  name = "rel_impl.bin_op"

  # TODO: could be restricted to only allow ints/floats
  lhs = OperandDef(DataType)
  rhs = OperandDef(DataType)

  # TODO: restrict to only *, +, - ...
  operator = AttributeDef(StringAttr)

  result = ResultDef(DataType)

  @staticmethod
  @builder
  def get(lhs: Operation, rhs: Operation, operator: str):
    return BinOp.create(operands=[lhs.result, rhs.result],
                        attributes={"operator": StringAttr.from_str(operator)},
                        result_types=[lhs.result.typ])


@irdl_op_definition
class Compare(Expression):
  """
  Returns `left` 'comparator' `right`.

  Example:

  '''
  ...
  %0 : !rel_impl.bool = rel_impl.compare(%a : !rel_impl.int32, %b : !rel_impl.int32) ["comparator" = "="]
  '''
  """
  name = "rel_impl.compare"

  left = OperandDef(DataType)
  right = OperandDef(DataType)
  comparator = AttributeDef(StringAttr)

  result = ResultDef(Boolean)

  @staticmethod
  @builder
  def get(left: Operation, right: Operation,
          comparator: StringAttr) -> 'Compare':
    return Compare.create(operands=[left.result, right.result],
                          attributes={"comparator": comparator},
                          result_types=[Boolean()])


class Yield(Expression):
  """
  Parent class of operations that gridge the gap from expressions back to
  operators by yielding the result of an expression to the encompassing
  operator.
  """
  ...


@irdl_op_definition
class YieldValue(Yield):
  """
  Bridges the gap from expressions back to operators by yielding the result of
  an expression to the encompassing operator. This `yield` is used for
  operations that want a value as the result of executing the region.

  Example:

  '''
  rel_impl.yield_value(%0 : !rel_impl.bool)
  '''
  """
  name = "rel_impl.yield_value"

  op = OperandDef(AnyAttr())

  @staticmethod
  @builder
  def get(op: Operation) -> 'YieldValue':
    return YieldValue.create(operands=[op.result])


@irdl_op_definition
class YieldTuple(Yield):
  """
  Bridges the gap from expressions back to operators by yielding the result of
  an expression to the encompassing operator. This `yield` is used for
  operations that want new tuples as the result of executing the region. This
  resulting tuple consists of the variadic operands of the `YieldTuple` in that
  order.

  Example:

  '''
  rel_impl.yield_tuple(%0 : !rel_impl.int32)
  '''
  """
  name = "rel_impl.yield_tuple"

  ops = VarOperandDef(AnyAttr())

  @staticmethod
  @builder
  def get(ops: list[Operation]) -> 'YieldTuple':
    return YieldTuple.create(operands=[o.result for o in ops])


@irdl_op_definition
class Literal(Expression):
  """
  Creates a literal with value `value`.

  Example:

  '''
  %0 : !rel_impl.int32 = rel_impl.literal() ["value" = 5 : i32]
  '''
  """
  name = "rel_impl.literal"

  # TODO: change IntegerAttr s.t. it can have type !rel.int32
  value = AttributeDef(AnyAttr())
  result = ResultDef(DataType)

  @staticmethod
  @builder
  def get(val: Attribute, res: DataType) -> 'Literal':
    return Literal.create(attributes={"value": val}, result_types=[res])


@irdl_op_definition
class And(Expression):
  """
  Computes the logical and of 'lhs' and 'rhs'.

  Example:

  '''
  %3 : !rel_impl.bool = rel_impl.and(%0 : !rel_impl.bool, %1 : !rel_impl.bool)
  '''
  """
  name = "rel_impl.and"

  lhs = OperandDef(Boolean)
  rhs = OperandDef(Boolean)

  result = ResultDef(Boolean)

  @staticmethod
  @builder
  def get(lhs: Operation, rhs: Operation) -> 'And':
    return And.create(operands=[lhs.result, rhs.result],
                      result_types=[Boolean()])


#===------------------------------------------------------------------------===#
# Operators
#===------------------------------------------------------------------------===#


class Operator(Operation):
  """
  Interface class for all Operators, i.e., operations that work on bags.
  """
  ...


@irdl_op_definition
class FullTableScanOp(Operator):
  """
  Performs a full table scan of the table `table_name` and produces a bag with
  the given schema. If the optional attribute cols is set, only the given
  columns are loaded. Otherwise, all columns are loaded.

  Example:

  '''
  %0 : rel_impl.bag<[!rel_impl.schema_element<"a", !rel_impl.int32>, !rel_impl.schema_element<"b", !rel_impl.int32>]> = rel_impl.full_table_scan() ["table_name" = "t"]
  %0 : rel_impl.bag<[!rel_impl.schema_element<"a", !rel_impl.int32>]> = rel_impl.full_table_scan() ["table_name" = "t", cols = ["a"]]
  '''
  """
  name = "rel_impl.full_table_scan"

  table_name = AttributeDef(StringAttr)
  cols = OptAttributeDef(ArrayAttr)
  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(name: str,
          result_type: Bag,
          cols: None | List[str] = None) -> 'FullTableScanOp':
    if cols:
      return FullTableScanOp.create(attributes={
          "table_name": StringAttr.from_str(name),
          "cols": ArrayAttr.from_list([StringAttr.from_str(s) for s in cols])
      },
                                    result_types=[result_type])
    return FullTableScanOp.create(
        attributes={"table_name": StringAttr.from_str(name)},
        result_types=[result_type])


@irdl_op_definition
class Select(Operator):
  """
  Selects all tuples of `input` according to the yielded expression in `predicates`.

  Example:

  '''
  rel_impl.select(%0: rel_impl.bag<[!rel_impl.schema_element<"id", !rel_impl.int32>]>) {
    ...
    rel_impl.yield(%3 : rel_impl.bool)
  }
  '''
  """
  name = "rel_impl.select"

  input = OperandDef(Bag)
  predicates = SingleBlockRegionDef()

  result = ResultDef(Bag)

  @builder
  @staticmethod
  def get(input: Operation, predicates: Region) -> 'Select':
    return Select.create(operands=[input.result],
                         regions=[predicates],
                         result_types=[input.results[0].typ])


@irdl_op_definition
class CartesianProduct(Operator):
  """
  Computes the Cartesian product of `left` with `right`.

  Example:

  '''
  %2 : !rel_impl.bag<...> = rel_impl.cartesian_product(%0 : !rel_impl.bag<...>, %1 : !rel_impl.bag<...>)
  '''
  """
  name = "rel_impl.cartesian_product"

  left = OperandDef(Bag)
  right = OperandDef(Bag)

  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(left: Operation, right: Operation) -> 'CartesianProduct':
    lhs_names = [e.elt_name.data for e in left.result.typ.schema.data]
    rhs_names = [e.elt_name.data for e in right.result.typ.schema.data]
    assert not any(i in lhs_names for i in rhs_names)
    return CartesianProduct.create(
        operands=[left.result, right.result],
        result_types=[
            Bag.get([e.elt_type for e in left.result.typ.schema.data] +
                    [e.elt_type for e in right.result.typ.schema.data],
                    lhs_names + rhs_names)
        ])


@irdl_op_definition
class Project(Operator):
  """
  Projects the input table s.t. every tuple is transformed to the yielded values
  in `projection`.

  Example:
  '''
  %1 : rel_impl.bag<[*schema_element1*]> = rel_impl.project(%0 :  rel_impl.bag<[*schema_element1*, *schema_element2*]>) {
    ^0(%2 : !rel_impl.tuple<[*schema_element1*, *schema_element2*])
      %3 : *type1* = rel_impl.index_by_name(%3 : ... ) ["col_name" = *name1*]
      rel_impl.yield(%3: *type1*)
  }
  '''

  """
  name = "rel_impl.project"

  projection = SingleBlockRegionDef()
  input = OperandDef(Bag)
  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(input: Operation, res_names: List[str], res_types: List[DataType],
          projection: Region) -> 'Project':
    return Project.create(operands=[input.result],
                          result_types=[Bag.get(res_types, res_names)],
                          regions=[projection])

  @staticmethod
  @builder
  def from_result_type(input: Operation, res_type: Bag,
                       projection: Region) -> 'Project':
    return Project.create(operands=[input.result],
                          result_types=[res_type],
                          regions=[projection])


@irdl_op_definition
class MergeSort(Operator):
  """
  Uses merge sort to sort the given input by the columns in `by`.

  Example:
  '''
  %{{.*}} : !rel_impl.bag<...> = rel_impl.merge_sort(%{{.*}} : !rel_impl.bag<...>) ["by" = [!rel_impl.order<"a", "desc">]]
  '''
  """
  name = "rel_impl.merge_sort"

  input = OperandDef(Bag)
  by = AttributeDef(ArrayAttr)

  result = ResultDef(Bag)

  @builder
  @staticmethod
  def get(input: Operation, by: list[str], order: list[str]) -> 'MergeSort':
    return MergeSort.build(
        operands=[input],
        attributes={
            "by":
                ArrayAttr.from_list([
                    Order([StringAttr.from_str(s),
                           StringAttr.from_str(o)]) for s, o in zip(by, order)
                ])
        },
        result_types=[input.result.typ])


@irdl_op_definition
class Aggregate(Operator):
  """
  Groups the table `input` by the columns in `by` by aggregating the ith element
  of `col_names` by the ith element of `functions`. If `by` is empty, this
  corresponds to the ungrouped aggregation. In the case of a `count(*)`, the
  respective element in `col_names` is `""` instead of a column name.


  Example:

  '''
  %0 : !rel_impl.bag<...> = rel_impl.aggregate(%0 : !rel_impl.bag<...>) ["col_names" = ["id"], "functions" = ["sum"], "by" = ["a"]]
  '''
  """
  name = "rel_impl.aggregate"

  input = OperandDef(Bag)
  col_names = AttributeDef(ArrayOfConstraint(StringAttr))
  functions = AttributeDef(ArrayOfConstraint(StringAttr))
  by = AttributeDef(ArrayOfConstraint(StringAttr))
  result = ResultDef(Bag)

  def verify_(self) -> None:
    if len(self.functions.data) != len(self.col_names.data):
      raise Exception(
          f"Number of functions and column names should match: {len(self.functions.data)} vs {len(self.col_names.data)}"
      )
    for f in self.functions.data:
      if not f.data in ["sum", "min", "max", "avg", "count", "count_distinct"]:
        raise Exception(f"function {f.data} is not a supported function")

  @builder
  @staticmethod
  def get(input: Operation, col_names: List[str], functions: List[str],
          res_names: List[str], by: List[str]) -> 'Aggregate':
    return Aggregate.create(
        operands=[input.result],
        attributes={
            "col_names":
                ArrayAttr.from_list([StringAttr.from_str(c) for c in col_names]
                                   ),
            "functions":
                ArrayAttr.from_list([StringAttr.from_str(f) for f in functions]
                                   ),
            "by":
                ArrayAttr.from_list([StringAttr.from_str(o) for o in by])
        },
        result_types=[
            Bag.get([
                Int64() if f in ["count", "count_distinct"] else
                input.result.typ.lookup_type_in_schema(n)
                for n, f in zip(col_names, functions)
            ], res_names)
        ])


@irdl_op_definition
class Limit(Operator):
  """
  Limits the number of tuples in `input` to `n` .

  Example:

  ```
  %0 : ... = rel_impl.limit(...) ["n" = 10 : !i64]
  ```
  """
  name = "rel_impl.limit"

  input = OperandDef(Bag)
  n = AttributeDef(IntegerAttr)

  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(table: Operation, n: int) -> 'Limit':
    return Limit.create(operands=[table.result],
                        attributes={"n": IntegerAttr.from_int_and_width(n, 64)},
                        result_types=[table.result.typ])


@dataclass
class RelImpl:
  ctx: MLContext

  def __post_init__(self: 'RelImpl'):
    self.ctx.register_attr(Bag)
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(Int64)
    self.ctx.register_attr(Nullable)
    self.ctx.register_attr(Decimal)
    self.ctx.register_attr(Timestamp)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Boolean)
    self.ctx.register_attr(SchemaElement)
    self.ctx.register_attr(Tuple)
    self.ctx.register_attr(Order)
    self.ctx.register_attr(Float64)

    self.ctx.register_op(Select)
    self.ctx.register_op(Limit)
    self.ctx.register_op(MergeSort)
    self.ctx.register_op(Project)
    self.ctx.register_op(CartesianProduct)
    self.ctx.register_op(Aggregate)
    self.ctx.register_op(FullTableScanOp)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Compare)
    self.ctx.register_op(IndexByName)
    self.ctx.register_op(YieldValue)
    self.ctx.register_op(YieldTuple)
    self.ctx.register_op(And)
    self.ctx.register_op(BinOp)
