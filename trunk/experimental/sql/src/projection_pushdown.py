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

# This file contains several rewrites that all compose to classic projection
# pushdown. Currently, this works in three steps and, hence, has three
# rewriters. The first one introduces a projection before any operator, the
# second one fuses subsequent projections, and the third one removes projections
# that are just identity mappings.


@dataclass
class ProjectionOptimizer(RewritePattern):

  # TODO: Move these functions to a utils file?
  def flatten(self, l: list[list[str]]) -> list[str]:
    return [elem for sublist in l for elem in sublist]

  def find_cols_in_expr(self, op: RelAlg.Expression) -> list[str]:
    """
    Returns the list of cols that are used in this expression or its children.
    """
    if isinstance(op, RelAlg.Column):
      return [op.col_name.data]
    else:
      return self.flatten([self.find_cols_in_expr(r.op) for r in op.regions])

  def find_cols_upstream(self, op: RelAlg.Operator) -> list[str]:
    """
    Returns the list of cols that are used in this operation or its parents. If
    the list contains None, a ModuleOp (so the output) was hit, showing that all
    columns need to be kept.
    """
    if isinstance(op, ModuleOp):
      return [None]
    if isinstance(op, RelAlg.Select):
      return self.flatten([
          self.find_cols_in_expr(o) for o in op.predicates.ops
      ]) + self.find_cols_upstream(op.parent_op())
    if isinstance(op, RelAlg.Project):
      return self.flatten(
          [self.find_cols_in_expr(o) for o in op.projections.ops])
    if isinstance(op, RelAlg.Aggregate):
      return [s.data for s in op.col_names.data if s.data != ""
             ] + [s.data for s in op.by.data]
    if isinstance(op, RelAlg.OrderBy):
      return [s.col.data for s in op.by.data] + self.find_cols_upstream(
          op.parent_op())
    if isinstance(op, RelAlg.Limit):
      return self.find_cols_upstream(op.parent_op())
    if isinstance(op, RelAlg.CartesianProduct):
      return self.find_cols_upstream(op.parent_op())

  def find_schema(self, op: RelAlg.Operator) -> list[str]:
    """
    Returns the list of cols that are the input schema of this operation.
    """
    if isinstance(op, RelAlg.Table):
      return [o.elt_name.data for o in op.schema.ops]
    if isinstance(op, RelAlg.Select):
      return self.find_schema(op.input.op)
    if isinstance(op, RelAlg.Project):
      return [s.data for s in op.names.data]
    if isinstance(op, RelAlg.Aggregate):
      return [s.data for s in op.col_names.data]
    if isinstance(op, RelAlg.CartesianProduct):
      return self.find_schema(op.left.op) + self.find_schema(op.right.op)
    if isinstance(op, RelAlg.OrderBy):
      return self.find_schema(op.input.op)
    if isinstance(op, RelAlg.Limit):
      return self.find_schema(op.input.op)


@dataclass
class ProjectionSimplifier(ProjectionOptimizer):

  # This rewriter simpifies projections such that they only project columns that
  # are actually needed upstream. This does not dot apply if there are any
  # renames.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Project, rewriter: PatternRewriter):
    cols = list(dict.fromkeys(self.find_cols_upstream(op.parent_op())))
    if None in cols or len(op.names.data) == len(cols):
      return

    res_names = [s.data for s in op.names.data]

    child_dict = dict(zip(res_names, op.projections.ops))

    new_proj = RelAlg.Project.get(
        Region.from_operation_list([op.input.op.clone()]),
        Region.from_operation_list(map(lambda x: child_dict[x].clone(), cols)),
        ArrayAttr.from_list([StringAttr.from_str(s) for s in cols]))
    rewriter.replace_matched_op(new_proj)


@dataclass
class ProjectionInference(ProjectionOptimizer):

  # This rewriter infers columns in front of all Operators (but projections and
  # tables) that map away all columns that aren't needed by this Operator or
  # further up the tree.

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Operator, rewriter: PatternRewriter):
    if isinstance(op, RelAlg.Project) or isinstance(
        op, RelAlg.Table) or isinstance(op, RelAlg.SchemaElement):
      return
    cols = list(dict.fromkeys(self.find_cols_upstream(op)))
    if None in cols:
      return

    if isinstance(op, RelAlg.CartesianProduct):
      left_cols = [c for c in cols if c in self.find_schema(op.left.op)]
      left_proj = RelAlg.Project.get(
          Region.from_operation_list([op.left.op.clone()]),
          Region.from_operation_list([RelAlg.Column.get(s) for s in left_cols]),
          ArrayAttr.from_list([StringAttr.from_str(s) for s in left_cols]))
      rewriter.replace_op(op.left.op, left_proj)
      right_cols = [c for c in cols if c in self.find_schema(op.right.op)]
      right_proj = RelAlg.Project.get(
          Region.from_operation_list([op.right.op.clone()]),
          Region.from_operation_list([RelAlg.Column.get(s) for s in right_cols
                                     ]),
          ArrayAttr.from_list([StringAttr.from_str(s) for s in right_cols]))
      rewriter.replace_op(op.right.op, right_proj)
    else:
      new_proj = RelAlg.Project.get(
          Region.from_operation_list([op.input.op.clone()]),
          Region.from_operation_list([RelAlg.Column.get(s) for s in cols]),
          ArrayAttr.from_list([StringAttr.from_str(s) for s in cols]))
      rewriter.replace_op(op.input.op, new_proj)


@dataclass
class ProjectionFuser(ProjectionOptimizer):

  # This rewriter fuses two subsequent projections. Currently, it only works on
  # renaming style projections, but it will be extended in the future.

  def replace_expr(self, op: RelAlg.Expression, map: dict[str,
                                                          RelAlg.Expression],
                   rewriter: PatternRewriter):
    if isinstance(op, RelAlg.Column):
      rewriter.replace_op(op, map[op.col_name.data].clone())
    else:
      for r in op.regions:
        self.replace_expr(r.op, map, rewriter)

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Project, rewriter: PatternRewriter):
    if not isinstance(op.input.op, RelAlg.Project):
      return

    child_expr = op.input.op.projections.ops
    child_res = [s.data for s in op.input.op.names.data]

    child_dict = dict(zip(child_res, op.input.op.projections.ops))
    new_projections = rewriter.move_region_contents_to_new_regions(
        op.projections)
    for o in new_projections.ops:
      self.replace_expr(o, child_dict, rewriter)
    rewriter.replace_matched_op(
        RelAlg.Project.get(
            rewriter.move_region_contents_to_new_regions(op.input.op.input),
            new_projections, op.names))


@dataclass
class IdentityProjectionRemover(ProjectionOptimizer):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Project, rewriter: PatternRewriter):

    # If any of the expressions in the projections list is not a Column
    # expression, this is not an IdentityProjection.
    # TODO: Maybe optimize simple expressions paterns (like add 0 or mult 1)
    if not all([isinstance(o, RelAlg.Column) for o in op.projections.ops]):
      return

    input_schema = self.find_schema(op.input.op)
    output_schema = self.flatten(
        [self.find_cols_in_expr(o) for o in op.projections.ops])

    if output_schema == [s.data for s in op.names.data
                        ] and input_schema == output_schema:
      new_op = op.input.blocks[0].detach_op(op.input.op)
      rewriter.replace_matched_op(new_op)


def projection_pushdown(ctx: MLContext, query: ModuleOp):
  simplify_projections_walker = PatternRewriteWalker(
      GreedyRewritePatternApplier([ProjectionSimplifier()]),
      walk_regions_first=False,
      apply_recursively=False,
      walk_reverse=False)
  simplify_projections_walker.rewrite_module(query)

  infer_projections_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjectionInference()]),
                                                  walk_regions_first=False,
                                                  apply_recursively=False,
                                                  walk_reverse=False)
  infer_projections_walker.rewrite_module(query)

  projection_fuser_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjectionFuser()]),
                                                 walk_regions_first=False,
                                                 apply_recursively=True,
                                                 walk_reverse=False)
  projection_fuser_walker.rewrite_module(query)

  identity_projection_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [IdentityProjectionRemover()]),
                                                    walk_regions_first=True,
                                                    apply_recursively=False,
                                                    walk_reverse=False)
  identity_projection_walker.rewrite_module(query)
