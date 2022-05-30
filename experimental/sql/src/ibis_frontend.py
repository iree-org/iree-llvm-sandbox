# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr
from xdsl.ir import Operation, MLContext, Region, Block
from typing import List, Type, Optional
from multipledispatch import dispatch

import ibis

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


@dispatch(ibis.expr.types.Expr)
def visit(  #type: ignore
    op: ibis.expr.types.Expr) -> Operation:
  return visit(op.op())


@dispatch(ibis.backends.pandas.client.PandasTable)
def visit(  #type: ignore
    op: ibis.backends.pandas.client.PandasTable) -> Operation:
  schema = visit_schema(op.schema)
  return id.PandasTable.get(op.name, schema)


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


@dispatch(ibis.expr.operations.logical.Equals)
def visit(  #type: ignore
    op: ibis.expr.operations.logical.Equals) -> Operation:
  left_reg = Region.from_operation_list([visit(op.left)])
  right_reg = Region.from_operation_list([visit(op.right)])
  return id.Equals.get(left_reg, right_reg)


@dispatch(ibis.expr.operations.generic.Literal)
def visit(  #type: ignore
    op: ibis.expr.operations.generic.Literal) -> Operation:
  return id.Literal.get(StringAttr.from_str(op.value),
                        convert_datatype(op.dtype))


@dispatch(ibis.expr.operations.reductions.Sum)
def visit(  #type: ignore
    op: ibis.expr.operations.reductions.Sum) -> Operation:
  arg = Region.from_operation_list([visit(op.arg)])
  return id.Sum.get(arg)


def ibis_to_xdsl(ctx: MLContext, query: ibis.expr.types.Expr) -> ModuleOp:
  return ModuleOp.build(regions=[Region.from_operation_list([visit(query)])])
