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

import dialects.ibis_dialect as id

# This file contains the translation from ibis to the ibis_dialect. This
# translation is implemented as a visitor-pattern with all functions being
# called using `visit(*some ibis node*), while the multipledispatch package
# ensures that the right function is actually called. In particular, this
# translation removes the expression layer from the ibis internal data
# structures and abstracts every ibis `op()` as an operation in the IR.


def convert_datatype(type_: ibis.expr.datatypes) -> id.DataType:
  if isinstance(type_, ibis.expr.datatypes.String):
    return id.String.get(1 if type_.nullable else 0)
  if isinstance(type_, ibis.expr.datatypes.Int32):
    return id.Int32()
  if isinstance(type_, ibis.expr.datatypes.Int64):
    return id.Int64()
  raise KeyError(f"Unknown datatype: {type(type_)}")


def convert_literal(literal) -> Attribute:
  if isinstance(literal, str):
    return StringAttr.from_str(literal)
  if isinstance(literal, np.int64):
    return IntegerAttr.from_int_and_width(literal, 64)
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
    ops.append(visit(op))
  return Region.from_operation_list(ops)


@dispatch(object)
def visit(op) -> Operation:
  raise Exception(f"conversion not yet implemented for {type(op)}")


@dispatch(ibis.expr.operations.numeric.Multiply)
def visit(op: ibis.expr.operations.numeric.Multiply) -> Operation:
  return id.Multiply.get(Region.from_operation_list([visit(op.left)]),
                         Region.from_operation_list([visit(op.right)]),
                         convert_datatype(op.output_type().keywords['dtype']))


@dispatch(ibis.expr.types.Expr)
def visit(  #type: ignore
    op: ibis.expr.types.Expr) -> Operation:
  return visit(op.op())


@dispatch(ibis.expr.operations.relations.UnboundTable)
def visit(  #type: ignore
    op: ibis.expr.operations.relations.UnboundTable) -> Operation:
  schema = visit_schema(op.schema)
  return id.UnboundTable.get(op.name, schema)


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
  return id.Selection.get(table, predicates, projections, names)


@dispatch(ibis.expr.operations.relations.Aggregation)
def visit(  #type: ignore
    op: ibis.expr.operations.relations.Aggregation) -> Operation:
  table = Region.from_operation_list([visit(op.table)])
  metrics = visit_ibis_expr_list(op.metrics)
  return id.Aggregation.get(table, metrics)


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


def ibis_to_xdsl(ctx: MLContext, query: ibis.expr.types.Expr) -> ModuleOp:
  return ModuleOp.build(regions=[Region.from_operation_list([visit(query)])])
