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

    # This rewrite expects that the projection only maps onto a subset of the
    # columns of the input. Therefore, we first check that only IndexByName is
    # present and then that there are equally as many cols that are yielded as
    # there are in the result.
    if not all([
        isinstance(o.op, RelImpl.IndexByName) for o in op.projection.ops[-1].ops
    ]):
      return

    assert isinstance(op.projection.ops[-1], RelImpl.YieldTuple)
    yielded_ssa_values = op.projection.ops[-1].ops

    col_names = [o.op.col_name.data for o in yielded_ssa_values]
    res_names = [s.elt_name.data for s in op.input.typ.schema.data]
    if len(col_names) == len(res_names):
      return

    col_types = [o.typ for o in yielded_ssa_values]

    new_op = RelImpl.FullTableScanOp.get(op.input.op.table_name.data,
                                         RelImpl.Bag.get(col_types, col_names),
                                         col_names)
    rewriter.replace_matched_op(new_op)

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
