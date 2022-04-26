# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr
from xdsl.ir import Operation, MLContext, Region, Block
from typing import List, Type, Optional
from multipledispatch import dispatch

import ibis
import ibis.expr.types
import ibis.expr.datatypes
import ibis.expr.operations.relations as rels
import ibis.expr.operations.generic as gen_types
from ibis.expr.operations.logical import Equals as EQ
import ibis.backends.pandas.client as PandasBackend

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


def visit_schema(schema: ibis.expr.schema.Schema) -> Region:  #type: ignore
  ops = []
  for n, t in zip(schema.names, schema.types):
    ops.append(id.SchemaElement.get(n, convert_datatype(t)))
  return Region.from_operation_list(ops)


def visit_ibis_expr_list(l: List[ibis.expr.types.Expr]) -> Region:
  ops = []
  for op in l:
    ops.append(visit(op))
  return Region.from_operation_list(ops)


@dispatch(ibis.expr.types.TableExpr)
def visit(  #type: ignore
    table: ibis.expr.types.TableExpr) -> Operation:  #type: ignore
  op = table.op()
  if isinstance(op, PandasBackend.PandasTable):
    schema = visit_schema(op.schema)
    new_op = id.PandasTable.get(op.name, schema)
    return new_op
  if isinstance(op, rels.Selection):
    table = Region.from_operation_list([visit(op.table)])
    # TODO: handle multiple predicates and projections
    predicate_ops = []
    for pred in op.predicates:
      predicate_ops.append(visit(pred))
    predicates = Region.from_operation_list(predicate_ops)
    projection_ops = []
    for proj in op.selections:
      projection_ops.append(visit(proj))
    projections = Region.from_operation_list(projection_ops)
    new_op = id.Selection.get(table, predicates, projections)
    return new_op
  if isinstance(op, rels.Aggregation):
    table = Region.from_operation_list([visit(op.table)])
    metrics = visit_ibis_expr_list(op.metrics)
    new_op = id.Aggregation.get(table, metrics)
    return new_op
  raise KeyError(f"Unknown tableExpr: {type(op)}")


@dispatch(ibis.expr.types.StringColumn)
def visit(  #type: ignore
    stringColumn: ibis.expr.types.StringColumn) -> Operation:
  op = stringColumn.op()
  if isinstance(op, gen_types.TableColumn):
    table = Region.from_operation_list([visit(op.table)])
    new_op = id.TableColumn.get(table, op.name)
    return new_op
  raise Exception(f"Unknown stringcolumn: {type(op)}")


@dispatch(ibis.expr.types.IntegerColumn)
def visit(  #type: ignore
    intColumn: ibis.expr.types.IntegerColumn) -> Operation:
  op = intColumn.op()
  if isinstance(op, gen_types.TableColumn):
    table = Region.from_operation_list([visit(op.table)])
    new_op = id.TableColumn.get(table, op.name)
    return new_op
  raise Exception(f"Unknown stringcolumn: {type(op)}")


@dispatch(ibis.expr.types.BooleanColumn)
def visit(  #type: ignore
    boolColumn: ibis.expr.types.BooleanColumn) -> Operation:
  op = boolColumn.op()
  if isinstance(op, gen_types.TableColumn):
    reg = Region.from_operation_list([visit(op.table)])
    return id.TableColumn.get(reg, op.name)
  if isinstance(op, EQ):
    left_reg = Region.from_operation_list([visit(op.left)])
    right_reg = Region.from_operation_list([visit(op.right)])
    new_op = id.Equals.get(left_reg, right_reg)
    return new_op
  raise Exception(f"Unknown booleancolumn: {type(op)}")


@dispatch(ibis.expr.types.StringScalar)
def visit(  #type: ignore
    strScalar: ibis.expr.types.StringScalar) -> Operation:
  op = strScalar.op()
  if isinstance(op, gen_types.Literal):
    new_op = id.Literal.get(StringAttr.from_str(op.value),
                            convert_datatype(op.dtype))
    return new_op
  raise Exception(f"Unknown stringscalar: {type(op)}")


@dispatch(ibis.expr.types.IntegerScalar)
def visit(  #type: ignore
    intScalar: ibis.expr.types.IntegerScalar) -> Operation:
  op = intScalar.op()
  if isinstance(op, ibis.expr.operations.reductions.Sum):
    arg = Region.from_operation_list([visit(op.arg)])
    new_op = id.Sum.get(arg)
    return new_op
  raise Exception(f"Unknown intScalar: {type(op)}")


def ibis_to_xdsl(ctx: MLContext, query: ibis.expr.types.Expr) -> ModuleOp:
  return ModuleOp.build(regions=[Region.from_operation_list([visit(query)])])
