# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Union, List, Optional
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute, OpResult
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

# This file contains the relational SSA dialect, a dialect that expresses
# relational queries using SSA. The dialect consists of two types of operations:
# Expressions and Operators. Operators work on Bags whereas expressions work on
# single values and are used to specify the exact behavior of operators. The
# Column and Yield Expressions bridge the gap to convert from one to the other.

#===------------------------------------------------------------------------===#
# Data types
#===------------------------------------------------------------------------===#


class DataType(ParametrizedAttribute):
  """
  Models a datatype in a relational SSA query.
  """
  name = "rel_ssa.datatype"


@irdl_attr_definition
class Int32(DataType):
  """
  Models a int32 type in a relational SSA query.

  Example:

  ```
  !rel_ssa.int32
  ```
  """
  name = "rel_ssa.int32"


@irdl_attr_definition
class Int64(DataType):
  """
  Models a int64 type in a relational SSA query.

  Example:

  ```
  !rel_ssa.int64
  ```
  """
  name = "rel_ssa.int64"


@irdl_attr_definition
class Decimal(DataType):
  """
  Models a decimal type in a relational SSA query with precision `prec` and scale `scale`.

  Example:

  ```
  !rel_ssa.decimal<4 : !i32, 2 : !i32>
  ```
  """
  name = "rel_ssa.decimal"

  prec: ParameterDef[IntegerAttr]
  scale: ParameterDef[IntegerAttr]


@irdl_attr_definition
class Timestamp(DataType):
  """
  Models a timestamp type in a relational SSA query.

  Example:

  ```
  !rel_ssa.timestamp
  ```
  """
  name = "rel_ssa.timestamp"


@irdl_attr_definition
class String(DataType):
  """
  Models a string type in a relational SSA query, that can be either nullable or
  not.

  Example:

  ```
  !rel_ssa.string
  ```
  """
  name = "rel_ssa.string"


@irdl_attr_definition
class Nullable(DataType):
  """
  Models a type that is nullable.

  Example:
  ```
  !rel_ssa.nullable<!rel_ssa.string>
  ```
  """
  name = "rel_ssa.nullable"

  type: ParameterDef[DataType]


#===------------------------------------------------------------------------===#
# Query types
#===------------------------------------------------------------------------===#


@irdl_attr_definition
class Boolean(ParametrizedAttribute):
  """
  Models a type that can either be true or false to, e.g., show whether a tuple
  is part of the result or not.

  Example:

  '''
  rel_ssa.select(%0: rel_ssa.bag<[!rel_ssa.int32]>) {
    ...
    rel_ssa.yield(%3 : rel_ssa.bool)
  }
  '''
  """
  name = "rel_ssa.bool"


@irdl_attr_definition
class SchemaElement(ParametrizedAttribute):
  """
  Models an element of a schema with name `elt_name` and type `elt_type`.

  Example:
  '''
  !rel_ssa.schema_element<"id", !rel_ssa.int32>
  '''
  """
  name = "rel_ssa.schema_element"

  elt_name: ParameterDef[StringAttr]
  elt_type: ParameterDef[DataType]

  @staticmethod
  @builder
  def get(name: str, type_: DataType) -> 'SchemaElement':
    return SchemaElement([StringAttr.from_str(name), type_])


@irdl_attr_definition
class Bag(ParametrizedAttribute):
  """
  Models a bag in a relational SSA query. The exact schema of the bag is part of
  the type itself.

  Example:

  '''
  !rel_ssa.bag<[!rel_ssa.schema_element<"id", !rel_ssa.int32>]>
  '''
  """
  name = "rel_ssa.bag"

  # TODO: add more verification
  # TODO: add DictionaryAttr to xDSL
  schema: ParameterDef[ArrayAttr[SchemaElement]]

  @staticmethod
  @builder
  def get(names: list[str], types: list[DataType]) -> 'Bag':
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
class Column(Expression):
  """
  References a specific column with name `col_name` and returns its value. The
  name 'col_name' must be part of the schema of an encompassing operator.

  Example:

  '''
  %0 : rel_ssa.int32 = rel_ssa.column() ["col_name" = "id"]
  '''
  """
  name = "rel_ssa.column"

  col_name = AttributeDef(StringAttr)
  result = ResultDef(DataType)

  @builder
  @staticmethod
  def get(name: str, res_type: DataType) -> 'Column':
    return Column.create(result_types=[res_type],
                         attributes={"col_name": StringAttr.from_str(name)})


@irdl_op_definition
class BinOp(Expression):
  """
  Models a binary operation of `lhs` `operator` `rhs`.

  Example:

  '''
  %1 : !rel_ssa.int32 = rel_ssa.bin_op(%0 : !rel_ssa.int32, %2 : !rel_ssa.int32) ["operator" = "*"]
  '''
  """
  name = "rel_ssa.bin_op"

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
  %0 : !rel_ssa.bool = rel_ssa.compare(%a : !rel_ssa.int32, %b : !rel_ssa.int32) ["comparator" = "="]
  '''
  """
  name = "rel_ssa.compare"

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
  rel_ssa.yield_value(%0 : !rel_impl.bool)
  '''
  """
  name = "rel_ssa.yield_value"

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
  rel_ssa.yield_tuple(%0 : !rel_impl.int32)
  '''
  """
  name = "rel_ssa.yield_tuple"

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
  %0 : !rel_ssa.int32 = rel_ssa.literal() ["value" = 5 : i32]
  '''
  """
  name = "rel_ssa.literal"

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
  Computes the logical and of 'lhs' and 'rhs.

  Example:

  '''
  %3 : !rel_ssa.bool = rel_ssa.and(%0 : !rel_ssa.bool, %1 : !rel_ssa.bool)
  '''
  """
  name = "rel_ssa.and"

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
  result = ResultDef(Bag)
  ...


@irdl_op_definition
class Table(Operator):
  """
  Defines a table with name `table_name`.

  Example:

  '''
  %0 : rel_ssa.bag<[!rel_ssa.int32]> = rel_ssa.table() ["table_name" = "t"]
  '''
  """
  name = "rel_ssa.table"

  table_name = AttributeDef(StringAttr)

  @staticmethod
  @builder
  def get(name: str, result_type: Attribute) -> 'Table':
    return Table.create(attributes={"table_name": StringAttr.from_str(name)},
                        result_types=[result_type])


@irdl_op_definition
class Project(Operator):
  """
  Projects the input table s.t. every tuple is transformed to the yielded values
  in `projection`.

  Example:
  '''
  %1 : rel_ssa.bag<[*schema_element1*]> = rel_ssa.project(%0 :  rel_ssa.bag<[*schema_element1*, *schema_element2*]>) {
    %2 : *type1* = rel_ssa.column() ["col_name" = *name1*]
    rel_ssa.yield(%2: *type1*)
  }
  '''

  """
  name = "rel_ssa.project"

  projection = SingleBlockRegionDef()
  input = OperandDef(Bag)
  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(input: Operation, res_names: List[str], res_types: List[DataType],
          projection: Region) -> 'Project':
    return Project.create(operands=[input.result],
                          result_types=[Bag.get(res_names, res_types)],
                          regions=[projection])

  @staticmethod
  @builder
  def from_result_type(input: Operation, res_type: Bag,
                       projection: Region) -> 'Project':
    return Project.create(operands=[input.result],
                          result_types=[res_type],
                          regions=[projection])


@irdl_op_definition
class Select(Operator):
  """
  Selects all tuples of `input` according to the yielded expression in `predicates`.

  Example:

  '''
  rel_ssa.select(%0: rel_ssa.bag<[!rel_ssa.int32]>) {
    ...
    rel_ssa.yield(%3 : rel_ssa.bool)
  }
  '''
  """
  name = "rel_ssa.select"

  input = OperandDef(Bag)
  predicates = SingleBlockRegionDef()

  @builder
  @staticmethod
  def get(input: Operation, predicates: Region) -> 'Select':
    return Select.create(operands=[input.result],
                         regions=[predicates],
                         result_types=[input.result.typ])


@irdl_op_definition
class CartesianProduct(Operator):
  """
  Computes the Cartesian product of operands `left` and `right`.

  Example:

  '''
  %2 : !rel_ssa.bag<...> = rel_ssa.cartesian_product(%0 : !rel_ssa.bag<...>, %1 : !rel_ssa.bag<...>)
  '''
  """
  name = "rel_ssa.cartesian_product"

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
            Bag.get(
                lhs_names + rhs_names,
                [e.elt_type for e in left.result.typ.schema.data] +
                [e.elt_type for e in right.result.typ.schema.data],
            )
        ])


@irdl_op_definition
class Aggregate(Operator):
  """
  Applies the ith function of `functions` to the ith column name of `col_names`
  of `input`. The ith resulting column has the same name as the ith input column
  to the uniqueness of names.


  Example:

  '''
  %0 : !rel_ssa.bag<[!rel_ssa.schema_element<"id", !rel_ssa.int32>]> = rel_ssa.aggregate(%0 : !rel_ssa.bag<[!rel_ssa.schema_element<"id", !rel_ssa.int32>]>) ["col_names" = ["id"], "functions" = ["sum"]] '''
  """
  name = "rel_ssa.aggregate"

  input = OperandDef(Bag)
  col_names = AttributeDef(ArrayOfConstraint(StringAttr))
  functions = AttributeDef(ArrayOfConstraint(StringAttr))
  result = ResultDef(Bag)

  def verify_(self) -> None:
    if len(self.functions.data) != len(self.col_names.data):
      raise Exception(
          f"Number of functions and column names should match: {len(self.functions.data)} vs {len(self.col_names.data)}"
      )
    for f in self.functions.data:
      if not f.data in ["sum"]:
        raise Exception(f"function {f.data} is not a supported function")

  @builder
  @staticmethod
  def get(input: Operation, col_names: List[str], functions: List[str],
          res_names: List[str]) -> 'Aggregate':
    return Aggregate.create(
        operands=[input.result],
        attributes={
            "col_names":
                ArrayAttr.from_list([StringAttr.from_str(c) for c in col_names]
                                   ),
            "functions":
                ArrayAttr.from_list([StringAttr.from_str(f) for f in functions])
        },
        result_types=[
            Bag.get(
                res_names,
                [input.result.typ.lookup_type_in_schema(n) for n in col_names])
        ])


@dataclass
class RelSSA:
  ctx: MLContext

  def __post_init__(self: 'RelSSA'):
    self.ctx.register_attr(Bag)
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(Int64)
    self.ctx.register_attr(Nullable)
    self.ctx.register_attr(Timestamp)
    self.ctx.register_attr(Decimal)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Boolean)
    self.ctx.register_attr(SchemaElement)

    self.ctx.register_op(Select)
    self.ctx.register_op(Table)
    self.ctx.register_op(Aggregate)
    self.ctx.register_op(Project)
    self.ctx.register_op(CartesianProduct)

    self.ctx.register_op(Literal)
    self.ctx.register_op(Compare)
    self.ctx.register_op(Column)
    self.ctx.register_op(YieldValue)
    self.ctx.register_op(YieldTuple)
    self.ctx.register_op(And)
    self.ctx.register_op(BinOp)
