# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.ir import Operation, MLContext, Region, Block, Attribute
from typing import List, Type, Optional
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr, IntegerType, TupleType
from xdsl.dialects.func import FuncOp, Return

from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, PatternRewriteWalker, PatternRewriter, op_type_rewrite_pattern

import dialects.rel_impl as RelImpl
import dialects.iterators as it

# This file contains the rewrite infrastructure to translate the relational
# implementation dialect to the iterators dialect. The current design has a
# parent class `RelImplRewriter` that contains functions used for several
# `Rewriter`s. All other `Rewriter`s inherit from that class.


@dataclass
class RelImplRewriter(RewritePattern):

  def convert_datatype(self, type_: RelImpl.DataType) -> Attribute:
    if isinstance(type_, RelImpl.Int32):
      return IntegerType.from_width(32)
    raise Exception(f"type conversion not yet implemented for {type(type_)}")

  def convert_bag(self, bag: RelImpl.Bag) -> it.Stream:
    types = [self.convert_datatype(s.elt_type) for s in bag.schema.data]
    return it.Stream.get(TupleType([ArrayAttr.from_list(types)]))


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#

#===------------------------------------------------------------------------===#
# Operators
#===------------------------------------------------------------------------===#


@dataclass
class PandasTableRewriter(RelImplRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelImpl.PandasTable,
                        rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        it.SampleInputOp.get(self.convert_bag(op.result.typ)))


@dataclass
class AggregateRewriter(RelImplRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelImpl.Aggregate, rewriter: PatternRewriter):
    rewriter.replace_matched_op(it.ReduceOp.get(op.input.op))


#===------------------------------------------------------------------------===#
# Conversion setup
#===------------------------------------------------------------------------===#


def impl_to_iterators(ctx: MLContext, query: ModuleOp):

  walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [PandasTableRewriter(), AggregateRewriter()]),
                                walk_regions_first=False,
                                apply_recursively=False,
                                walk_reverse=False)
  walker.rewrite_module(query)
  # Adding the sink
  query.body.blocks[0].add_op(
      it.SinkOp.get(query.body.blocks[0].ops[-1].results[0]))
  # Adding the return
  query.body.blocks[0].add_op(Return.get())
  # Wrapping everything into a main function
  f = FuncOp.from_region("main", [], [],
                         Region.from_block_list([query.body.detach_block(0)]))
  query.body.add_block(Block.from_ops([f]))
