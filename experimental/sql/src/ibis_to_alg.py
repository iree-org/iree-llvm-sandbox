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
      return RelAlg.String.get(type_.nullable)
    if isinstance(type_, ibis.Int32):
      return RelAlg.Int32()
    raise Exception(
        f"datatype conversion not yet implemented for {type(type_)}")


@dataclass
class SchemaElementRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.SchemaElement,
                        rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.SchemaElement.get(op.elt_name.data,
                                 self.convert_datatype(op.elt_type)))


@dataclass
class PandasTableRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.PandasTable, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.PandasTable.get(
            op.table_name.data,
            rewriter.move_region_contents_to_new_regions(op.schema)))


@dataclass
class LiteralRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Literal, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Literal.get(op.val, self.convert_datatype(op.type)))


@dataclass
class EqualsRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Equals, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Compare.get(
            "=", rewriter.move_region_contents_to_new_regions(op.left),
            rewriter.move_region_contents_to_new_regions(op.right)))


@dataclass
class TableColumnRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.TableColumn, rewriter: PatternRewriter):
    rewriter.replace_matched_op(RelAlg.Column.get(op.col_name.data))


@dataclass
class SelectionRewriter(IbisRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Selection, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelAlg.Select.get(
            rewriter.move_region_contents_to_new_regions(op.table),
            rewriter.move_region_contents_to_new_regions(op.predicates)))


@dataclass
class AggregationRewriter(IbisRewriter):

  def get_col_name_and_function(self, metric_op: Operation) -> Tuple[str, str]:
    if isinstance(metric_op, ibis.Sum):
      return "sum", metric_op.arg.op.attributes["col_name"].data
    raise Exception(
        f"aggregation function not yet implemented {type(metric_op)}")

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: ibis.Aggregation, rewriter: PatternRewriter):
    functions, col_names = map(
        list, zip(*[self.get_col_name_and_function(o) for o in op.metrics.ops]))

    rewriter.replace_matched_op(
        RelAlg.Aggregate.get(
            rewriter.move_region_contents_to_new_regions(op.table), col_names,
            functions))


def ibis_to_alg(ctx: MLContext, query: ModuleOp):
  walker = PatternRewriteWalker(GreedyRewritePatternApplier([
      PandasTableRewriter(),
      SchemaElementRewriter(),
      SelectionRewriter(),
      EqualsRewriter(),
      TableColumnRewriter(),
      AggregationRewriter(),
      LiteralRewriter()
  ]),
                                walk_regions_first=False,
                                apply_recursively=True,
                                walk_reverse=True)
  walker.rewrite_module(query)
