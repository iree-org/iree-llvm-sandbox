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

import dialects.rel_impl as RelImpl

# This file contains rewrites that aim at fusing a table directly following
# projection, that projects on a subset of the columns in the table.


@dataclass
class ProjScanFuser(RewritePattern):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelImpl.Project, rewriter: PatternRewriter):
    if not isinstance(op.input.op, RelImpl.FullTableScanOp):
      return

    assert isinstance(op.projection.ops[-1], RelImpl.YieldTuple)
    yielded_ssa_values = op.projection.ops[-1].ops

    # Get the list of used columns. The dict magic removes duplicates.
    used_cols = list(
        dict.fromkeys([
            o.col_name.data
            for o in op.projection.ops
            if isinstance(o, RelImpl.IndexByName)
        ]))
    input_names = [s.elt_name.data for s in op.input.typ.schema.data]

    # If all the input cols are used, there is nothing to fuse.
    if len(used_cols) == len(input_names):
      return

    col_types = [o.typ for o in yielded_ssa_values]

    new_op = RelImpl.FullTableScanOp.get(op.input.op.table_name.data,
                                         RelImpl.Bag.get(col_types, used_cols),
                                         used_cols)

    if all([
        isinstance(o.op, RelImpl.IndexByName) for o in op.projection.ops[-1].ops
    ]):
      # If the projection is simple, replace the projection by a partial load.
      rewriter.replace_matched_op(new_op)
    else:
      # For a complex projection, replace the projection with a projection
      # reading from the partial load.
      new_reg = rewriter.move_region_contents_to_new_regions(op.projection)
      rewriter.modify_block_argument_type(
          new_reg.blocks[0].args[0],
          RelImpl.Tuple.get(
              col_types,
              [s.elt_name.data for s in new_op.result.typ.schema.data]))
      rewriter.insert_op_before_matched_op(new_op)
      rewriter.replace_matched_op(
          RelImpl.Project.from_result_type(new_op, op.result.typ, new_reg))

    if len(op.input.uses) == 0:
      # TODO: remove that safe_erase is False, when we have some kind of dead code
      # elimination.
      rewriter.erase_op(op.input.op, safe_erase=False)


def fuse_proj_into_scan(ctx: MLContext, query: ModuleOp):
  fuse_proj_table_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjScanFuser()]),
                                                walk_regions_first=False,
                                                apply_recursively=False,
                                                walk_reverse=False)
  fuse_proj_table_walker.rewrite_module(query)
