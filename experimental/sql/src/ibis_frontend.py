# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr
from xdsl.ir import Operation, MLContext, Region, Block, Attribute
from typing import List, Type, Optional
from multipledispatch import dispatch

import ibis
import numpy as np

from decimal import Decimal

import dialects.ibis_dialect as id

# This file contains the translation from ibis to the ibis_dialect. This
# translation is implemented as a visitor-pattern with all functions being
# called using `visit(*some ibis node*), while the multipledispatch package
# ensures that the right function is actually called. In particular, this
# translation removes the expression layer from the ibis internal data
# structures and abstracts every ibis `op()` as an operation in the IR.


def convert_datatype(type_: ibis.expr.datatypes) -> id.DataType:
  ret_type = None
  if isinstance(type_, ibis.expr.datatypes.String):
    ret_type = id.String()
  elif isinstance(type_, ibis.expr.datatypes.Int32):
    ret_type = id.Int32()
  elif isinstance(type_, ibis.expr.datatypes.Int64):
    ret_type = id.Int64()
  elif isinstance(type_, ibis.expr.datatypes.Float64):
    ret_type = id.Float64()
  elif isinstance(type_, ibis.expr.datatypes.Timestamp):
    ret_type = id.Timestamp()
  elif isinstance(type_, ibis.expr.datatypes.Decimal):
    ret_type = id.Decimal([
        IntegerAttr.from_int_and_width(type_.precision, 32),
        IntegerAttr.from_int_and_width(type_.scale, 32)
    ])
  else:
    raise Exception(
        f"datatype conversion not yet implemented for {type(type_)}")
  return id.Nullable([ret_type]) if type_.nullable else ret_type


def convert_literal(literal) -> Attribute:
  if isinstance(literal, str):
    return StringAttr.from_str(literal)
  if isinstance(literal, float):
    # TODO: This is a workaround until xdsl versions are released properly.
    return IntegerAttr.from_int_and_width(int(literal), 64)
  if isinstance(literal, int):
    # np.int64 are parsed as int by ibis
    return IntegerAttr.from_int_and_width(literal, 64)
  if isinstance(literal, Decimal):
    return StringAttr.from_str(str(literal))
  raise Exception(f"literal conversion not yet implemented for {type(literal)}")


# The first two functions work on multiple parts of the ibis tree, so they
# return `Region`s and cannot be written using multipledispatch.
def visit_schema(schema: ibis.expr.schema.Schema) -> Region:
  ops = []
  for n, t in zip(schema.names, schema.types):
    ops.append(id.SchemaElement.get(n, convert_datatype(t)))
  return Region.from_operation_list(ops)


def visit_ibis_expr_list(l: List[ibis.expr.types.Expr]) -> Region:
  ops = []
  for op in l:
    # TODO: This is a hack that rewrites top-level Ands. To do this properly,
    # and handle all adds, first model And in ibis, then either rewrite this to
    # a version using an optimization on the ibis dialect or model And in
    # rel_alg and lower to that.
    if isinstance(op.op(), ibis.expr.operations.logical.And):
      ops.append(visit(op.op().left))
      ops.append(visit(op.op().right))
    else:
      ops.append(visit(op))
  return Region.from_operation_list(ops)


@dispatch(object)
def visit(op) -> Operation:
  raise Exception(f"conversion not yet implemented for {type(op)}")


@dispatch(ibis.expr.operations.numeric.Multiply)
def visit(  # type: ignore
    op: ibis.expr.operations.numeric.Multiply) -> Operation:
  return id.Multiply.get(Region.from_operation_list([visit(op.left)]),
                         Region.from_operation_list([visit(op.right)]),
                         convert_datatype(op.output_dtype()))


@dispatch(ibis.expr.operations.numeric.Subtract)
def visit(op: ibis.expr.operations.numeric.Subtract):
  return id.Subtract.get(Region.from_operation_list([visit(op.left)]),
                         Region.from_operation_list([visit(op.right)]),
                         convert_datatype(op.output_dtype()))


@dispatch(ibis.expr.operations.numeric.Add)
def visit(op: ibis.expr.operations.numeric.Add):
  return id.Add.get(Region.from_operation_list([visit(op.left)]),
                    Region.from_operation_list([visit(op.right)]),
                    convert_datatype(op.output_dtype()))


@dispatch(ibis.expr.operations.numeric.Divide)
def visit(op: ibis.expr.operations.numeric.Divide):
  return id.Divide.get(Region.from_operation_list([visit(op.left)]),
                       Region.from_operation_list([visit(op.right)]),
                       convert_datatype(op.output_dtype()))


@dispatch(ibis.expr.operations.core.Alias)
def visit(  #type: ignore
    op: ibis.expr.operations.core.Alias) -> Operation:
  return visit(op.arg.op())


@dispatch(ibis.expr.types.Expr)
def visit(  #type: ignore
    op: ibis.expr.types.Expr) -> Operation:
  return visit(op.op())


@dispatch(ibis.expr.operations.relations.UnboundTable)
def visit(  #type: ignore
    op: ibis.expr.operations.relations.UnboundTable) -> Operation:
  schema = visit_schema(op.schema)
  return id.UnboundTable.get(op.name, schema)


@dispatch(ibis.expr.operations.relations.InnerJoin)
def visit(op: ibis.expr.operations.relations.InnerJoin) -> Operation:
  cart_prod = id.CartesianProduct.get(
      Region.from_operation_list([visit(op.left)]),
      Region.from_operation_list([visit(op.right)]))
  if op.predicates:
    return id.Selection.get(Region.from_operation_list([cart_prod]),
                            visit_ibis_expr_list(op.predicates),
                            Region.from_operation_list([]),
                            Region.from_operation_list([]), [])
  return cart_prod


@dispatch(ibis.expr.operations.relations.Limit)
def visit(op: ibis.expr.operations.relations.Limit) -> Operation:
  return id.Limit.get(Region.from_operation_list([visit(op.table)]), op.n)


@dispatch(ibis.expr.operations.relations.Selection)
def visit(  #type: ignore
    op: ibis.expr.operations.relations.Selection) -> Operation:
  assert (op.inputs[0] is op.table)
  names = []
  if len(op.inputs) > 0:
    names = [n.get_name() for n in op.inputs[1]]
  table = Region.from_operation_list([visit(op.table)])
  predicates = visit_ibis_expr_list(op.predicates)
  projections = visit_ibis_expr_list(op.selections)
  sort_keys = visit_ibis_expr_list(op.sort_keys)
  return id.Selection.get(table, predicates, projections, sort_keys, names)


@dispatch(ibis.expr.operations.logical.Between)
def visit(op):
  arg = Region.from_operation_list([visit(op.arg)])
  lower_bound = Region.from_operation_list([visit(op.lower_bound)])
  upper_bound = Region.from_operation_list([visit(op.upper_bound)])
  return id.Between.get(arg, lower_bound, upper_bound)


@dispatch(ibis.expr.operations.sortkeys.SortKey)
def visit(op: ibis.expr.operations.sortkeys.SortKey) -> Operation:
  return id.SortKey.get(Region.from_operation_list([visit(op.expr)]),
                        op.ascending)


@dispatch(ibis.expr.operations.relations.Aggregation)
def visit(  #type: ignore
    op: ibis.expr.operations.relations.Aggregation) -> Operation:
  table = Region.from_operation_list([visit(op.table)])
  metrics = visit_ibis_expr_list(op.metrics)
  by = visit_ibis_expr_list(op.by)
  names = []
  if len(op.inputs) > 0:
    names = [n.get_name() for n in op.inputs[1]]
  return id.Aggregation.get(table, metrics, by, names)


@dispatch(ibis.expr.operations.generic.TableColumn)
def visit(  #type: ignore
    op: ibis.expr.operations.generic.TableColumn) -> Operation:
  table = Region.from_operation_list([visit(op.table)])
  return id.TableColumn.get(table, op.name)


def create_logical_op(op: ibis.expr.operations.Comparison,
                      _class: Operation) -> Operation:
  left_reg = Region.from_operation_list([visit(op.left)])
  right_reg = Region.from_operation_list([visit(op.right)])
  return _class.get(left_reg, right_reg)


@dispatch(ibis.expr.operations.logical.Equals)
def visit(  #type: ignore
    op: ibis.expr.operations.logical.Equals) -> Operation:
  return create_logical_op(op, id.Equals)


@dispatch(ibis.expr.operations.logical.GreaterEqual)
def visit(  #type: ignore
    op: ibis.expr.operations.logical.GreaterEqual) -> Operation:
  return create_logical_op(op, id.GreaterEqual)


@dispatch(ibis.expr.operations.logical.Greater)
def visit(  #type: ignore
    op: ibis.expr.operations.logical.Greater) -> Operation:
  return create_logical_op(op, id.GreaterThan)


@dispatch(ibis.expr.operations.logical.LessEqual)
def visit(  #type: ignore
    op: ibis.expr.operations.logical.LessEqual) -> Operation:
  return create_logical_op(op, id.LessEqual)


@dispatch(ibis.expr.operations.logical.Less)
def visit(  #type: ignore
    op: ibis.expr.operations.logical.Less) -> Operation:
  return create_logical_op(op, id.LessThan)


@dispatch(ibis.expr.operations.generic.Literal)
def visit(  #type: ignore
    op: ibis.expr.operations.generic.Literal) -> Operation:
  return id.Literal.get(convert_literal(op.value), convert_datatype(op.dtype))


@dispatch(ibis.expr.operations.reductions.Sum)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.Sum) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.Sum.get(arg)


@dispatch(ibis.expr.operations.reductions.Mean)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.Mean) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.Mean.get(arg)


@dispatch(ibis.expr.operations.reductions.Max)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.Max) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.Max.get(arg)


@dispatch(ibis.expr.operations.reductions.Min)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.Min) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.Min.get(arg)


@dispatch(ibis.expr.operations.reductions.Count)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.Count) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.Count.get(arg)


@dispatch(ibis.expr.operations.reductions.CountDistinct)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.CountDistinct) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.CountDistinct.get(arg)


def ibis_to_xdsl(ctx: MLContext, query: ibis.expr.types.Expr) -> ModuleOp:
  return ModuleOp.build(regions=[Region.from_operation_list([visit(query)])])
