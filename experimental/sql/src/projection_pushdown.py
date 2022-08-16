# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.ir import Operation, MLContext, Region, Block
from typing import List, Type, Optional
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr

from xdsl.pattern_rewriter import (RewritePattern, GreedyRewritePatternApplier,
                                   PatternRewriteWalker, PatternRewriter,
                                   op_type_rewrite_pattern)

import dialects.rel_alg as RelAlg


@dataclass
class ProjectionPushdownRewriter(RewritePattern):

  def find_cols_in_expr(self, op: RelAlg.Expression) -> list[str]:
    """
    Returns the list of cols that are used in this operation or its children.
    """
    if isinstance(op, RelAlg.BinOp):
      return self.find_cols_in_expr(op.lhs.op) + self.find_cols_in_expr(
          op.rhs.op)
    if isinstance(op, RelAlg.Column):
      return [op.col_name.data]
    if isinstance(op, RelAlg.Compare):
      return self.find_cols_in_expr(op.left.op) + self.find_cols_in_expr(
          op.right.op)
    return []

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Project, rewriter: PatternRewriter):
    if not isinstance(op.input.op, RelAlg.Select):
      return

    select_cols = [
        elem for sublist in (
            self.find_cols_in_expr(o) for o in op.input.op.predicates.ops)
        for elem in sublist
    ]

    project_cols = [
        elem for sublist in (self.find_cols_in_expr(o)
                             for o in op.projections.ops) for elem in sublist
    ]

    result_cols = [s.data for s in op.names.data]

    # TODO: handle renames of columns

    # TODO: think about what happens on aggregate. Aggregate is very similar to
    # project in what it does with schemas

    # Check whether all cols in the select are contained in the project.
    if all(item in project_cols for item in select_cols):
      new_proj = RelAlg.Project.get(
          rewriter.move_region_contents_to_new_regions(op.input.op.input),
          Region.from_operation_list(
              [RelAlg.Column.get(n) for n in project_cols]),
          ArrayAttr.from_list([StringAttr.from_str(s) for s in project_cols]))
      new_sel = RelAlg.Select.get(
          Region.from_operation_list([new_proj]),
          rewriter.move_region_contents_to_new_regions(op.input.op.predicates))

      # If the output of the select has the same schema as the initial project,
      # we don't need a second project.
      if set(result_cols) == set(select_cols):
        rewriter.replace_matched_op(new_sel)
      else:
        new_proj2 = RelAlg.Project.get(
            Region.from_operation_list([new_sel]),
            rewriter.move_region_contents_to_new_regions(op.projections),
            ArrayAttr.from_list([StringAttr.from_str(s) for s in result_cols]))
        rewriter.replace_matched_op(new_proj2)


def projection_pushdown(ctx: MLContext, query: ModuleOp):
  walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjectionPushdownRewriter()]),
                                walk_regions_first=False,
                                apply_recursively=False,
                                walk_reverse=False)
  walker.rewrite_module(query)
