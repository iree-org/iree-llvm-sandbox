# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.ir import Operation, MLContext, Region, Block
from typing import List, Type, Optional, Tuple
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr

from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, PatternRewriteWalker, PatternRewriter, op_type_rewrite_pattern

import dialects.ibis_dialect as ibis
import dialects.rel_alg as RelAlg

# This file defines several `Rewriter`s that translate ibis dialect operations to
# the corresponding constructs in the relational algebra dialect. The
# `IbisRewriter` defines the parent class that contains all functions used by
# several specific `Rewriter`s (like convert_datatype). All specific `Rewriter`s
# inherit from the IbisRewriter.


@dataclass
class IbisRewriter(RewritePattern):

  def convert_datatype(self, type_: ibis.DataType) -> RelAlg.DataType:
    if isinstance(type_, ibis.String):
      return RelAlg.String()
    if isinstance(type_, ibis.Int32):
      return RelAlg.Int32()
    if isinstance(type_, ibis.Int64):
      return RelAlg.Int64()
    if isinstance(type_, ibis.Float64):
      return RelAlg.Float64()
    if isinstance(type_, ibis.Timestamp):
      return RelAlg.Timestamp()
    if isinstance(type_, ibis.Decimal):
      return RelAlg.Decimal([
          IntegerAttr.from_int_and_width(type_.prec.value.data,
                                         type_.prec.typ.width.data),
          IntegerAttr.from_int_and_width(type_.scale.value.data,
                                         type_.scale.typ.width.data)
      ])
    if isinstance(type_, ibis.Nullable):
      return RelAlg.Nullable([self.convert_datatype(type_.datatype)])
    raise Exception(
        f"datatype conversion not yet implemented for {type(type_)}")


@dataclass
class SchemaElementRewriter(IbisRewriter):
  # This is a simple 1-1 rewrite.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.SchemaElement,
                        rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.SchemaElement.get(op.elt_name.data,
                                 self.convert_datatype(op.elt_type)))


@dataclass
class UnboundTableRewriter(IbisRewriter):
  # This is a simple 1-1 rewrite.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.UnboundTable, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Table.get(
            op.table_name.data,
            rewriter.move_region_contents_to_new_regions(op.schema)))


@dataclass
class LiteralRewriter(IbisRewriter):
  # This is a simple 1-1 rewrite.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Literal, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Literal.get(op.val, self.convert_datatype(op.type)))


@dataclass
class LimitRewriter(IbisRewriter):
  # This is a simple 1-1 rewrite.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Limit, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Limit.get(rewriter.move_region_contents_to_new_regions(op.table),
                         op.n.value.data))


@dataclass
class EqualsRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Equals, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            "=", rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class GreaterEqualRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.GreaterEqual, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            ">=", rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class GreaterThanRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.GreaterThan, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            ">", rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class LessThanRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.LessThan, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            "<", rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class LessEqualRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.LessEqual, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            "<=", rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class TableColumnRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.TableColumn, rewriter: PatternRewriter):
    rewriter.replace_matched_op(RelAlg.Column.get(op.col_name.data))


@dataclass
class SelectionRewriter(IbisRewriter):
  # This is a simple 1-1 rewrite.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Selection, rewriter: PatternRewriter):
    assert len(op.predicates.ops) == 0 or len(op.projections.ops) == 0
    if len(op.predicates.ops) > 0:
      rewriter.replace_matched_op(
          RelAlg.Select.get(
              rewriter.move_region_contents_to_new_regions(op.table),
              rewriter.move_region_contents_to_new_regions(op.predicates)))
    elif len(op.projections.ops) > 0:
      rewriter.replace_matched_op(
          RelAlg.Project.get(
              rewriter.move_region_contents_to_new_regions(op.table),
              rewriter.move_region_contents_to_new_regions(op.projections),
              op.names))
    else:
      assert all(
          [isinstance(o.expr.op, ibis.TableColumn) for o in op.sort_keys.ops])
      rewriter.replace_matched_op(
          RelAlg.OrderBy.get(
              rewriter.move_region_contents_to_new_regions(op.table),
              [o.expr.op.col_name.data for o in op.sort_keys.ops],
              [o.order.data for o in op.sort_keys.ops]))


@dataclass
class CartesianProductRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.CartesianProduct,
                        rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.CartesianProduct.get(
            rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class AggregationRewriter(IbisRewriter):

  def get_col_name_and_function(self, metric_op: Operation) -> Tuple[str, str]:
    if isinstance(metric_op, ibis.Sum):
      return "sum", metric_op.arg.op.attributes["col_name"].data
    if isinstance(metric_op, ibis.Min):
      return "min", metric_op.arg.op.attributes["col_name"].data
    if isinstance(metric_op, ibis.Max):
      return "max", metric_op.arg.op.attributes["col_name"].data
    if isinstance(metric_op, ibis.Mean):
      return "avg", metric_op.arg.op.attributes["col_name"].data
    if isinstance(metric_op, ibis.CountDistinct):
      return "count_distinct", metric_op.arg.op.attributes["col_name"].data
    if isinstance(metric_op, ibis.Count):
      if isinstance(metric_op.arg.op, ibis.TableColumn):
        return "count", metric_op.arg.op.attributes["col_name"].data
      else:
        return "count", ""
    raise Exception(
        f"aggregation function not yet implemented {type(metric_op)}")

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Aggregation, rewriter: PatternRewriter):
    functions, col_names = map(
        list, zip(*[self.get_col_name_and_function(o) for o in op.metrics.ops]))

    assert all([isinstance(o, ibis.TableColumn) for o in op.by.ops])

    by = [o.col_name.data for o in op.by.ops]

    rewriter.replace_matched_op(
        RelAlg.Aggregate.get(
            rewriter.move_region_contents_to_new_regions(op.table), col_names,
            functions, [r.data for r in op.names.data], by))


@dataclass
class MultiplyRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Multiply, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.BinOp.get(rewriter.move_region_contents_to_new_regions(op.lhs),
                         rewriter.move_region_contents_to_new_regions(op.rhs),
                         "*"))


@dataclass
class DivideRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Divide, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.BinOp.get(rewriter.move_region_contents_to_new_regions(op.lhs),
                         rewriter.move_region_contents_to_new_regions(op.rhs),
                         "/"))


@dataclass
class SubtractRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Subtract, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.BinOp.get(rewriter.move_region_contents_to_new_regions(op.lhs),
                         rewriter.move_region_contents_to_new_regions(op.rhs),
                         "-"))


@dataclass
class AddRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Add, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.BinOp.get(rewriter.move_region_contents_to_new_regions(op.lhs),
                         rewriter.move_region_contents_to_new_regions(op.rhs),
                         "+"))


@dataclass
class BetweenRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Between, rewriter: PatternRewriter):
    rewriter.insert_op_before_matched_op(
        RelAlg.Compare.get(
            "<=", rewriter.move_region_contents_to_new_regions(op.lower_bound),
            Region.from_operation_list([o.clone() for o in op.arg.ops])))
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            "<=", rewriter.move_region_contents_to_new_regions(op.arg),
            rewriter.move_region_contents_to_new_regions(op.upper_bound)))


def ibis_to_alg(ctx: MLContext, query: ModuleOp):
  walker = PatternRewriteWalker(GreedyRewritePatternApplier([
      UnboundTableRewriter(),
      SchemaElementRewriter(),
      SelectionRewriter(),
      CartesianProductRewriter(),
      EqualsRewriter(),
      GreaterEqualRewriter(),
      GreaterThanRewriter(),
      LessEqualRewriter(),
      LessThanRewriter(),
      TableColumnRewriter(),
      AggregationRewriter(),
      MultiplyRewriter(),
      SubtractRewriter(),
      DivideRewriter(),
      AddRewriter(),
      LimitRewriter(),
      BetweenRewriter(),
      LiteralRewriter()
  ]),
                                walk_regions_first=False,
                                apply_recursively=True,
                                walk_reverse=True)
  walker.rewrite_module(query)
