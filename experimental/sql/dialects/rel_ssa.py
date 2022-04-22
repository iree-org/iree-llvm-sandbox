# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Union, List
from xdsl.ir import Block, Region, Operation, SSAValue, ParametrizedAttribute, Data, MLContext, Attribute
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


@irdl_attr_definition
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
class String(DataType):
  """
  Models a string type in a relational SSA query, that can be either nullable or
  not.

  Example:

  ```
  !rel_ssa.string<0: !i1>
  ```
  """
  name = "rel_ssa.string"

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
  !rel_ssa.bool
  '''
  """
  name = "rel_ssa.bool"


@irdl_attr_definition
class Bag(ParametrizedAttribute):
  """
  Models a bag in a relational SSA query. The exact schema of the bag is part of
  the type itself.

  Example:

  '''
  !rel_ssa.bag<[!rel_ssa.int32]>
  '''
  """
  name = "rel_ssa.bag"

  schema = ParameterDef(ArrayOfConstraint(DataType))

  @staticmethod
  @builder
  def get(types: list[DataType]) -> 'Bag':
    return Bag([ArrayAttr.from_list(types)])  #type: ignore


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
    return Compare.build(operands=[left, right],
                         attributes={"comparator": comparator},
                         result_types=[Boolean])


@irdl_op_definition
class Yield(Expression):
  """
  Bridges the gap from expressions back to operators by yielding the result of
  an expression to the encompassing operator.

  Example:

  '''
  rel_ssa.yield(%0 : !rel_ssa.bool)
  '''
  """
  name = "rel_ssa.yield"

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
  Defines a table with name `table_name`.

  Example:

  '''
  %0 : rel_ssa.bag<[!rel_ssa.int32]> = rel_ssa.pandas_table() ["table_name" = "t"]
  '''
  """
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

  result = ResultDef(Bag)

  @builder
  @staticmethod
  def get(table: Operation, predicates: Region) -> 'Select':
    return Select.build(operands=[table],
                        regions=[predicates],
                        result_types=[Bag()])


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
