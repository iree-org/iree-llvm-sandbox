# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.ir import Operation, MLContext, Region, Block
from typing import List, Type, Optional
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr

from xdsl.pattern_rewriter import RewritePattern, GreedyRewritePatternApplier, PatternRewriteWalker, PatternRewriter, op_type_rewrite_pattern

import dialects.rel_ssa as RelSSA
import dialects.rel_impl as RelImpl

# This file contains the rewrite infrastructure to translate the relational SSA
# dialect to the relational implementation dialect. The current design has a
# parent class `RelSSARewriter` that contains functions used for several
# `Rewriter`s. All other `Rewriter`s inherit from that class.


@dataclass
class RelSSARewriter(RewritePattern):

  def convert_datatype(self, type_: RelSSA.DataType) -> RelImpl.DataType:
    if isinstance(type_, RelSSA.String):
      return RelImpl.String()
    if isinstance(type_, RelSSA.Int32):
      return RelImpl.Int32()
    if isinstance(type_, RelSSA.Int64):
      return RelImpl.Int64()
    if isinstance(type_, RelSSA.Float64):
      return RelImpl.Float64()
    if isinstance(type_, RelSSA.Decimal):
      return RelImpl.Decimal([
          IntegerAttr.from_int_and_width(type_.prec.value.data,
                                         type_.prec.typ.width.data),
          IntegerAttr.from_int_and_width(type_.scale.value.data,
                                         type_.scale.typ.width.data)
      ])
    if isinstance(type_, RelSSA.Timestamp):
      return RelImpl.Timestamp()
    if isinstance(type_, RelSSA.Nullable):
      return RelImpl.Nullable([self.convert_datatype(type_.type)])
    raise Exception(
        f"datatype conversion not yet implemented for {type(type_)}")

  def convert_bag(self, bag: RelSSA.Bag) -> RelImpl.Bag:
    names = [s.elt_name.data for s in bag.schema.data]
    types = [self.convert_datatype(s.elt_type) for s in bag.schema.data]
    return RelImpl.Bag.get(types, names)

  def create_tuple_of_bag(self, bag: RelImpl.Bag) -> RelImpl.Tuple:
    names = [s.elt_name.data for s in bag.schema.data]
    types = [s.elt_type for s in bag.schema.data]
    return RelImpl.Tuple.get(types, names)


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#


@dataclass
class LiteralRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Literal, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        [RelImpl.Literal.get(op.value, self.convert_datatype(op.result.typ))])


@dataclass
class ColumnRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Column, rewriter: PatternRewriter):
    # NOTE: This relies on the fact that all input columns have different names.
    assert len(op.parent.args) == 1 or not any([
        i in [s.elt_name.data
              for s in op.parent.args[0].typ.schema.data]
        for i in [s.elt_name.data for s in op.parent.args[1].typ.schema.data]
    ])
    tuple_arg = None
    for arg in op.parent.args:
      if op.col_name.data in [s.elt_name.data for s in arg.typ.schema.data]:
        tuple_arg = arg
        break
    assert tuple_arg
    rewriter.replace_matched_op([
        RelImpl.IndexByName.get(op.col_name.data, tuple_arg,
                                self.convert_datatype(op.result.typ))
    ])


@dataclass
class CompareRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Compare, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        [RelImpl.Compare.get(op.left.op, op.right.op, op.comparator)])


@dataclass
class AndRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.And, rewriter: PatternRewriter):
    rewriter.replace_matched_op([RelImpl.And.get(op.lhs.op, op.rhs.op)])


@dataclass
class BinOpRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.BinOp, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        [RelImpl.BinOp.get(op.lhs.op, op.rhs.op, op.operator.data)])


@dataclass
class YieldTupleRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.YieldTuple, rewriter: PatternRewriter):
    # TODO: This is a hack to circumvent the FrozenList. Could this be done cleaner?
    rewriter.replace_matched_op(
        [RelImpl.YieldTuple.get([o.op for o in op.operands])])


@dataclass
class YieldValueRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.YieldValue, rewriter: PatternRewriter):
    # TODO: This is a hack to circumvent the FrozenList. Could this be done cleaner?
    rewriter.replace_matched_op([RelImpl.YieldValue.get(op.op.op)])


#===------------------------------------------------------------------------===#
# Operators
#===------------------------------------------------------------------------===#


@dataclass
class OrderByRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.OrderBy, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelImpl.MergeSort.get(op.input.op, [o.col.data for o in op.by.data],
                              [o.order.data for o in op.by.data]))


@dataclass
class ProjectRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Project, rewriter: PatternRewriter):
    predicates = Region.from_block_list(
        [Block.from_arg_types([self.create_tuple_of_bag(op.input.typ)])])
    # The following loop moves the operations of (the old) op.predicates to the
    # predicates region of the new operation.
    for o in op.projection.blocks[0].ops:
      op.projection.blocks[0].detach_op(o)
      predicates.blocks[0].add_op(o)
    rewriter.replace_matched_op(
        RelImpl.Project.from_result_type(op.input.op,
                                         self.convert_bag(op.result.typ),
                                         predicates))


@dataclass
class CartesianProductRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.CartesianProduct,
                        rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelImpl.CartesianProduct.get(op.left.op, op.right.op))


@dataclass
class SelectRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Select, rewriter: PatternRewriter):
    predicates = Region.from_block_list(
        [Block.from_arg_types([self.create_tuple_of_bag(op.input.typ)])])
    # The following loop moves the operations of (the old) op.predicates to the
    # predicates region of the new operation.
    for o in op.predicates.blocks[0].ops:
      op.predicates.blocks[0].detach_op(o)
      predicates.blocks[0].add_op(o)
    rewriter.replace_matched_op(RelImpl.Select.get(op.input.op, predicates))


@dataclass
class TableRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Table, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelImpl.FullTableScanOp.get(op.table_name.data,
                                    self.convert_bag(op.result.typ)))


@dataclass
class AggregateRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Aggregate, rewriter: PatternRewriter):
    rewriter.replace_matched_op(
        RelImpl.Aggregate.get(
            op.input.op, [c.data for c in op.col_names.data],
            [f.data for f in op.functions.data],
            [s.elt_name.data for s in op.result.typ.schema.data],
            [o.data for o in op.by.data]))


@dataclass
class LimitRewriter(RelSSARewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Limit, rewriter: PatternRewriter):
    rewriter.replace_matched_op(RelImpl.Limit.get(op.input.op, op.n.value.data))


#===------------------------------------------------------------------------===#
# Conversion setup
#===------------------------------------------------------------------------===#


def ssa_to_impl(ctx: MLContext, query: ModuleOp):

  walker = PatternRewriteWalker(GreedyRewritePatternApplier([
      TableRewriter(),
      SelectRewriter(),
      AggregateRewriter(),
      LiteralRewriter(),
      ColumnRewriter(),
      CompareRewriter(),
      YieldTupleRewriter(),
      YieldValueRewriter(),
      ProjectRewriter(),
      AndRewriter(),
      CartesianProductRewriter(),
      OrderByRewriter(),
      LimitRewriter(),
      BinOpRewriter()
  ]),
                                walk_regions_first=False,
                                apply_recursively=True,
                                walk_reverse=False)
  walker.rewrite_module(query)
