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

# This file contains several rewrites on the rel_alg level, all of which concern
# projections. Currently, there are two rewriters. The first one introduces a
# projection before any operator, the second one removes projetions that are
# just identity mappings.


@dataclass
class ProjectionOptimizer(RewritePattern):

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
      return [s.data for s in op.col_names.data]

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
    if cols.__contains__(None):
      return

    new_proj = RelAlg.Project.get(
        rewriter.move_region_contents_to_new_regions(op.input),
        Region.from_operation_list([RelAlg.Column.get(s) for s in cols]),
        ArrayAttr.from_list([StringAttr.from_str(s) for s in cols]))

    # TODO: I mainly need this pattern match to call the proper builder. This
    # could maybe be done nicer.
    if isinstance(op, RelAlg.Select):
      rewriter.replace_matched_op(
          RelAlg.Select.get(
              Region.from_operation_list([new_proj]),
              rewriter.move_region_contents_to_new_regions(op.predicates)))
    elif isinstance(op, RelAlg.Aggregate):
      rewriter.replace_matched_op(
          RelAlg.Aggregate.get(Region.from_operation_list([new_proj]),
                               [s.data for s in op.col_names.data],
                               [s.data for s in op.functions.data],
                               [s.data for s in op.res_names.data]))


@dataclass
class ProjectionFuser(ProjectionOptimizer):

  def rename_expr(self, op: RelAlg.Expression, map: dict[str,
                                                         str]) -> Operation:
    if isinstance(op, RelAlg.Column):
      return RelAlg.Column.get(map[op.col_name.data])
    if isinstance(op, RelAlg.Multiply):
      return RelAlg.Multiply.get(
          Region.from_operation_list([self.rename_expr(op.lhs, map)]),
          Region.from_operation_list([self.rename_expr(op.rhs, map)]))
    if isinstance(op, RelAlg.Literal):
      return RelAlg.Literal.get(op.val, op.type)
    if isinstance(op, RelAlg.Compare):
      return RelAlg.Multiply.get(
          op.comparator.data,
          Region.from_operation_list([self.rename_expr(op.left, map)]),
          Region.from_operation_list([self.rename_expr(op.right, map)]))

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Project, rewriter: PatternRewriter):
    if not isinstance(op.input.op, RelAlg.Project):
      return

    child_cols = self.flatten(
        [self.find_cols_in_expr(o) for o in op.input.op.projections.ops])
    child_res = [s.data for s in op.input.op.names.data]
    # If the child projection simply projects onto a subset, we can just infer
    # its column mapping and apply it on the parents projection.
    if len(child_cols) == len(child_res) and all(
        [isinstance(o, RelAlg.Column) for o in op.input.op.projections.ops]):
      child_dict = dict(zip(child_cols, child_res))

      rewriter.replace_matched_op(
          RelAlg.Project.get(
              rewriter.move_region_contents_to_new_regions(op.input.op.input),
              Region.from_operation_list([
                  self.rename_expr(o, child_dict) for o in op.projections.ops
              ]), op.names))


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


def optimize_projections(ctx: MLContext, query: ModuleOp):
  infer_projections_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjectionInference()]),
                                                  walk_regions_first=False,
                                                  apply_recursively=False,
                                                  walk_reverse=False)
  infer_projections_walker.rewrite_module(query)

  projection_fuser_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjectionFuser()]),
                                                 walk_regions_first=False,
                                                 apply_recursively=False,
                                                 walk_reverse=False)
  projection_fuser_walker.rewrite_module(query)

  identity_projection_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [IdentityProjectionRemover()]),
                                                    walk_regions_first=True,
                                                    apply_recursively=False,
                                                    walk_reverse=False)
  identity_projection_walker.rewrite_module(query)
