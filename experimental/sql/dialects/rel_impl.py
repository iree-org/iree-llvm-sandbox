# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Union, List
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
from xdsl.dialects.builtin import StringAttr, ArrayAttr, ArrayOfConstraint, IntegerAttr
from xdsl.irdl import AttributeDef, OperandDef, ResultDef, RegionDef, SingleBlockRegionDef, irdl_attr_definition, irdl_op_definition, ParameterDef, AnyAttr, VarOperandDef, builder

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


@irdl_attr_definition
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
class String(DataType):
  """
  Models a string type in a relational implementation query, that can be either
  nullable or not.

  Example:

  ```
  !rel_impl.string<0: !i1>
  ```
  """
  name = "rel_impl.string"

  # TODO: redefine nullable as a property of all fields
  nullable = ParameterDef(IntegerAttr)

  @staticmethod
  @builder
  def get(val: Union[int, IntegerAttr]) -> 'String':
    if isinstance(val, IntegerAttr):
      return String([val])
    return String([IntegerAttr.from_int_and_width(val, 1)])


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

  elt_name = ParameterDef(StringAttr)
  elt_type = ParameterDef(DataType)

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

  schema = ParameterDef(ArrayOfConstraint(SchemaElement))

  @staticmethod
  @builder
  def get(types: list[DataType], names: list[str]) -> 'Bag':
    schema_elts = [SchemaElement.get(n, t) for n, t in zip(names, types)]
    return Bag([ArrayAttr.from_list(schema_elts)])


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

  schema = ParameterDef(ArrayOfConstraint(SchemaElement))

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
    return IndexByName.build(operands=[tuple],
                             result_types=[res_type],
                             attributes={"col_name": StringAttr.from_str(name)})


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
    return Compare.build(operands=[left, right],
                         attributes={"comparator": comparator},
                         result_types=[Boolean()])


@irdl_op_definition
class Yield(Expression):
  """
  Bridges the gap from expressions back to operators by yielding the result of
  an expression to the encompassing operator.

  Example:

  '''
  rel_impl.yield(%0 : !rel_impl.bool)
  '''
  """
  name = "rel_impl.yield"

  ops = VarOperandDef(AnyAttr())

  @staticmethod
  @builder
  def get(ops: list[Operation]) -> 'Yield':
    return Yield.build(operands=[ops])


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
    return Literal.build(attributes={"value": val}, result_types=[res])


#===------------------------------------------------------------------------===#
# Operators
#===------------------------------------------------------------------------===#


class Operator(Operation):
  """
  Interface class for all Operators, i.e., operations that work on bags.
  """
  ...


@irdl_op_definition
class PandasTable(Operator):
  """
  Defines a table with name `table_name` and the schema defined in the result_type.

  Example:

  '''
  %0 : rel_impl.bag<[!rel_impl.int32]> = rel_impl.pandas_table() ["table_name" = "t"]
  '''
  """
  name = "rel_impl.pandas_table"

  table_name = AttributeDef(StringAttr)
  result = ResultDef(Bag)

  @staticmethod
  @builder
  def get(name: str, result_type: Bag) -> 'PandasTable':
    return PandasTable.build(
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
    return Select.build(operands=[input],
                        regions=[predicates],
                        result_types=[input.results[0].typ])


@dataclass
class RelImpl:
  ctx: MLContext

  def __post_init__(self: 'RelImpl'):
    self.ctx.register_attr(Bag)
    self.ctx.register_attr(DataType)
    self.ctx.register_attr(Int32)
    self.ctx.register_attr(String)
    self.ctx.register_attr(Boolean)
    self.ctx.register_attr(SchemaElement)
    self.ctx.register_attr(Tuple)

    self.ctx.register_op(Select)
    self.ctx.register_op(PandasTable)
    self.ctx.register_op(Literal)
    self.ctx.register_op(Compare)
    self.ctx.register_op(IndexByName)
    self.ctx.register_op(Yield)
