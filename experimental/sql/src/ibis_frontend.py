# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from modulefinder import Module
from unittest import result
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr
from xdsl.ir import Operation, MLContext, Region, Block
from typing import List, Type, Optional

import ibis
import ibis.expr.types
import ibis.expr.datatypes
import ibis.expr.operations.relations as rels
import ibis.expr.operations.generic as gen_types
from ibis.expr.operations.logical import Equals as EQ
import ibis.backends.pandas.client as PandasBackend

import dialects.ibis_dialect as id

# This file contains the translation from ibis to the ibis_dialect.
# It is built in a visitor-style, as this is one of the best designs
# to crawl trees in my experience. The entry-point is the `visit`-
# function just below. It dispatches to the appropriate translator.


@dataclass
class NodeVisitor:
  ctx: MLContext

  def visit(self, node: ibis.expr.types.Expr) -> List[Operation]:
    if isinstance(node, ibis.expr.types.TableExpr):
      return self.visit_TableExpr(node)
    if isinstance(node, ibis.expr.types.StringColumn):
      return self.visit_StringColumn(node)
    if isinstance(node, ibis.expr.types.BooleanColumn):
      return self.visit_BooleanColumn(node)
    if isinstance(node, ibis.expr.types.StringScalar):
      return self.visit_StringScalar(node)
    raise KeyError(f"Unknown nodetype {type(node)}")

  def convert_datatype(self, type_: ibis.expr.datatypes) -> id.DataType:
    if isinstance(type_, ibis.expr.datatypes.String):
      return id.String.get(1 if type_.nullable else 0)
    if isinstance(type_, ibis.expr.datatypes.Int32):
      return id.int32()
    if isinstance(type_, ibis.expr.datatypes.Float64):
      return id.float64()
    raise KeyError(f"Unknown datatype: {type(type_)}")

  def visit_Schema(self, schema: ibis.expr.schema.Schema) -> Region:
    ops = []
    for n, t in zip(schema.names, schema.types):
      ops.append(id.SchemaElement.get(n, self.convert_datatype(t)))
    return Region.from_operation_list(ops)

  def visit_TableExpr(self,
                      table: ibis.expr.types.TableExpr) -> List[Operation]:
    op = table.op()
    if isinstance(op, PandasBackend.PandasTable):
      schema = self.visit_Schema(op.schema)
      new_op = id.PandasTable.get(op.name, schema)
      yield_op = id.Yield.get([new_op])
      return [new_op, yield_op]
    if isinstance(op, rels.Selection):
      table = Region.from_operation_list(self.visit(op.table))
      # TODO: handle mulitple predicates
      predicates = Region.from_operation_list(self.visit(op.predicates[0]))
      new_op = id.Selection.get(table, predicates)
      yield_op = id.Yield.get([new_op])
      return [new_op, yield_op]
    raise KeyError(f"Unknown tableExpr: {type(op)}")

  def visit_StringColumn(
      self, stringColumn: ibis.expr.types.StringColumn) -> List[Operation]:
    op = stringColumn.op()
    if isinstance(op, gen_types.TableColumn):
      table = Region.from_operation_list(self.visit(op.table))
      new_op = id.TableColumn.get(table, op.name, id.StringColumn())
      yield_op = id.Yield.get([new_op])
      return [new_op, yield_op]
    raise Exception(f"Unknown stringcolumn: {type(op)}")

  def visit_BooleanColumn(
      self, stringColumn: ibis.expr.types.BooleanColumn) -> List[Operation]:
    op = stringColumn.op()
    if isinstance(op, gen_types.TableColumn):
      reg = Region.from_operation_list(self.visit(op.table))
      return [
          id.TableColumn.build(
              attributes={"col_name": StringAttr.from_str(op.name)},
              regions=[reg])
      ]
    if isinstance(op, EQ):
      left_reg = Region.from_operation_list(self.visit(op.left))
      right_reg = Region.from_operation_list(self.visit(op.right))
      new_op = id.Equals.get(left_reg, right_reg)
      yield_op = id.Yield.get([new_op])
      return [new_op, yield_op]
    raise Exception(f"Unknown booleancolumn: {type(op)}")

  def visit_StringScalar(
      self, strScalar: ibis.expr.types.StringScalar) -> List[Operation]:
    op = strScalar.op()
    if isinstance(op, gen_types.Literal):
      new_op = id.Literal.get(StringAttr.from_str(op.value),
                              self.convert_datatype(op.dtype))
      yield_op = id.Yield.get([new_op])
      return [new_op, yield_op]
    raise Exception(f"Unknown stringscalar: {type(op)}")


def ibis_to_xdsl(ctx: MLContext, query: ibis.expr.types.Expr) -> ModuleOp:
  return ModuleOp.build(
      regions=[Region.from_operation_list(NodeVisitor(ctx).visit(query))])
