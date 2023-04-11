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
import dialects.rel_ssa as RelSSA

# This file contains the rewrite infrastructure to translate the relational
# algebra dialect to the relational SSA dialect. The current design has a parent
# class `RelAlgRewriter` that contains functions used for several `Rewriter`s.
# All other `Rewriter`s inherit from that class. To introduce SSA, the passes
# insert the children (and the their whole block), that should become operands,
# before the matched op, then use the `added_operations_before` functionality of
# the rewriter to link the operands to the new operation.


@dataclass
class RelAlgRewriter(RewritePattern):

  def convert_datatype(self, type_: RelAlg.DataType) -> RelSSA.DataType:
    if isinstance(type_, RelAlg.String):
      return RelSSA.String()
    if isinstance(type_, RelAlg.Int32):
      return RelSSA.Int32()
    if isinstance(type_, RelAlg.Int64):
      return RelSSA.Int64()
    if isinstance(type_, RelAlg.Float64):
      return RelSSA.Float64()
    if isinstance(type_, RelAlg.Decimal):
      return RelSSA.Decimal([
          IntegerAttr.from_int_and_width(type_.prec.value.data,
                                         type_.prec.typ.width.data),
          IntegerAttr.from_int_and_width(type_.scale.value.data,
                                         type_.scale.typ.width.data)
      ])
    if isinstance(type_, RelAlg.Timestamp):
      return RelSSA.Timestamp()
    if isinstance(type_, RelAlg.Nullable):
      return RelSSA.Nullable([self.convert_datatype(type_.type)])
    raise Exception(
        f"datatype conversion not yet implemented for {type(type_)}")

  def find_type_in_parent_operator_inputs(
      self, name: str, op: Operation) -> Optional[RelSSA.DataType]:
    """
    Crawls through all parent_ops until reaching either a ModuleOp, in which
    case the lookup failed or reaching an operator with one or multiple input
    bags, that the type can be looked up in.
    """
    parent_op = op.parent_op()
    while (parent_op and not isinstance(parent_op, ModuleOp)):
      if isinstance(parent_op, RelSSA.Operator):
        for o in parent_op.operands:
          type_ = o.typ.lookup_type_in_schema(name)
          if type_:
            return type_
        raise Exception(f"element not found in parent schema: {name}")
      parent_op = parent_op.parent_op()
    raise Exception(f"element not found in parent schema: {name}")


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#
"""
All expression rewriters implicitly assume that the last operation in a block is
the one to be yielded.
"""


@dataclass
class LiteralRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Literal, rewriter: PatternRewriter):
    new_op = RelSSA.Literal.get(op.val, self.convert_datatype(op.type))
    rewriter.replace_matched_op([new_op, RelSSA.YieldTuple.get([new_op])])


@dataclass
class ColumnRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Column, rewriter: PatternRewriter):
    res_type = self.find_type_in_parent_operator_inputs(op.col_name.data, op)
    new_op = RelSSA.Column.get(op.col_name.data, res_type)
    rewriter.replace_matched_op([new_op, RelSSA.YieldTuple.get([new_op])])


@dataclass
class CompareRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Compare, rewriter: PatternRewriter):
    # Remove the yield and inline the rest of the block.
    rewriter.erase_op(op.left.blocks[0].ops[-1])
    rewriter.inline_block_before_matched_op(op.left.blocks[0])
    left = rewriter.added_operations_before[-1]

    rewriter.erase_op(op.right.blocks[0].ops[-1])
    rewriter.inline_block_before_matched_op(op.right.blocks[0])
    right = rewriter.added_operations_before[-1]

    new_op = RelSSA.Compare.get(left, right, op.comparator)
    rewriter.replace_matched_op([new_op, RelSSA.YieldValue.get(new_op)])


@dataclass
class BinOpRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.BinOp, rewriter: PatternRewriter):
    # Remove the yield and inline the rest of the block.
    rewriter.erase_op(op.lhs.blocks[0].ops[-1])
    rewriter.inline_block_before_matched_op(op.lhs.blocks[0])
    left = rewriter.added_operations_before[-1]

    rewriter.erase_op(op.rhs.blocks[0].ops[-1])
    rewriter.inline_block_before_matched_op(op.rhs.blocks[0])
    right = rewriter.added_operations_before[-1]

    # TODO: Make Decimals change their prec and scale on certain operations.
    new_op = RelSSA.BinOp.get(left, right, op.operator.data)
    rewriter.replace_matched_op([new_op, RelSSA.YieldTuple.get([new_op])])


#===------------------------------------------------------------------------===#
# Operators
#===------------------------------------------------------------------------===#


@dataclass
class CartesianProductRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.CartesianProduct,
                        rewriter: PatternRewriter):
    rewriter.inline_block_before_matched_op(op.left.blocks[0])
    left = rewriter.added_operations_before[-1]
    rewriter.inline_block_before_matched_op(op.right.blocks[0])
    right = rewriter.added_operations_before[-1]

    rewriter.insert_op_before_matched_op(
        RelSSA.CartesianProduct.get(left, right))
    rewriter.erase_matched_op()


@dataclass
class OrderByRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.OrderBy, rewriter: PatternRewriter):
    rewriter.inline_block_before_matched_op(op.input.blocks[0])
    input = rewriter.added_operations_before[-1]

    rewriter.insert_op_before_matched_op(
        RelSSA.OrderBy.get(input, [o.col.data for o in op.by.data],
                           [o.order.data for o in op.by.data]))
    rewriter.erase_matched_op()


@dataclass
class ProjectRewriter(RelAlgRewriter):

  # TODO: This could be implemented in a more natural way using Analysis Passes in MLIR.
  def find_type_of_expression(self, op: RelAlg.Expression,
                              input_bag: RelSSA.Bag) -> RelSSA.DataType:
    if isinstance(op, RelAlg.Column):
      return input_bag.lookup_type_in_schema(op.col_name.data)
    if isinstance(op, RelAlg.Literal):
      return op.type
    if isinstance(op, RelAlg.BinOp):
      # This uses the assumption that the lhs and rhs have the same type.
      return self.find_type_of_expression(op.lhs.op, input_bag)
    raise Exception(f"expression conversion not yet implemented for {type(op)}")

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Project, rewriter: PatternRewriter):
    rewriter.inline_block_before_matched_op(op.input)
    input_bag = rewriter.added_operations_before[-1].result.typ
    rewriter.insert_op_before_matched_op(
        RelSSA.Project.get(
            rewriter.added_operations_before[-1],
            [n.data for n in op.names.data], [
                self.find_type_of_expression(op, input_bag)
                for op in op.projections.ops
            ], rewriter.move_region_contents_to_new_regions(op.projections)))
    rewriter.erase_matched_op()


@dataclass
class ProjectYieldCombiner(RewritePattern):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Project, rewriter: PatternRewriter):
    yielded_ops = []
    for operation in op.projection.ops:
      if isinstance(operation, RelSSA.YieldTuple):
        yielded_ops.extend([o.op for o in operation.ops])
        op.projection.blocks[0].erase_op(operation)
    new_region = rewriter.move_region_contents_to_new_regions(op.projection)
    new_region.blocks[0].add_op(RelSSA.YieldTuple.get(yielded_ops))
    rewriter.replace_matched_op(
        RelSSA.Project.from_result_type(op.input.op, op.result.typ, new_region))


@dataclass
class SelectYieldCombiner(RewritePattern):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelSSA.Select, rewriter: PatternRewriter):
    yielded_ops = []
    for operation in op.predicates.ops:
      if isinstance(operation, RelSSA.YieldValue):
        yielded_ops.append(operation.op.op)
        op.predicates.blocks[0].erase_op(operation)
    new_region = rewriter.move_region_contents_to_new_regions(op.predicates)
    while (len(yielded_ops) > 1):
      new_and = RelSSA.And.get(yielded_ops.pop(), yielded_ops.pop())
      yielded_ops.append(new_and)
      new_region.blocks[0].add_op(new_and)
    new_region.blocks[0].add_op(RelSSA.YieldValue.get(yielded_ops[0]))
    rewriter.replace_matched_op(RelSSA.Select.get(op.input.op, new_region))


@dataclass
class SelectRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Select, rewriter: PatternRewriter):
    rewriter.inline_block_before_matched_op(op.input)
    predicates = rewriter.move_region_contents_to_new_regions(op.predicates)
    rewriter.insert_op_before_matched_op(
        RelSSA.Select.get(rewriter.added_operations_before[-1], predicates))
    rewriter.erase_matched_op()


@dataclass
class TableRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Table, rewriter: PatternRewriter):
    schema_names = [s.elt_name.data for s in op.schema.ops]
    schema_types = [self.convert_datatype(s.elt_type) for s in op.schema.ops]
    result_type = RelSSA.Bag.get(schema_names, schema_types)
    new_op = RelSSA.Table.get(op.table_name.data, result_type)
    rewriter.insert_op_before_matched_op(new_op)
    rewriter.erase_matched_op()


@dataclass
class AggregateRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Aggregate, rewriter: PatternRewriter):
    rewriter.inline_block_before_matched_op(op.input.blocks[0])
    rewriter.insert_op_before_matched_op([
        RelSSA.Aggregate.get(rewriter.added_operations_before[-1],
                             [c.data for c in op.col_names.data],
                             [f.data for f in op.functions.data],
                             [r.data for r in op.res_names.data],
                             [b.data for b in op.by.data])
    ])
    rewriter.erase_matched_op()


@dataclass
class LimitRewriter(RelAlgRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelAlg.Limit, rewriter: PatternRewriter):
    rewriter.inline_block_before_matched_op(op.input.blocks[0])
    rewriter.insert_op_before_matched_op([
        RelSSA.Limit.get(rewriter.added_operations_before[-1], op.n.value.data)
    ])
    rewriter.erase_matched_op()


#===------------------------------------------------------------------------===#
# Conversion setup
#===------------------------------------------------------------------------===#

# This pass first rewrites operators and then expressions, since the schema of
# the encompassing operator needs to be known in order to know the type of an
# expression.


def alg_to_ssa(ctx: MLContext, query: ModuleOp):
  operator_walker = PatternRewriteWalker(GreedyRewritePatternApplier([
      TableRewriter(),
      SelectRewriter(),
      AggregateRewriter(),
      ProjectRewriter(),
      OrderByRewriter(),
      LimitRewriter(),
      CartesianProductRewriter()
  ]),
                                         walk_regions_first=True,
                                         apply_recursively=True,
                                         walk_reverse=False)
  operator_walker.rewrite_module(query)
  expression_walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [LiteralRewriter(),
       ColumnRewriter(),
       CompareRewriter(),
       BinOpRewriter()]),
                                           walk_regions_first=True,
                                           apply_recursively=True,
                                           walk_reverse=False)
  expression_walker.rewrite_module(query)
  yield_combiner = PatternRewriteWalker(GreedyRewritePatternApplier(
      [ProjectYieldCombiner(), SelectYieldCombiner()]),
                                        walk_regions_first=True,
                                        apply_recursively=False,
                                        walk_reverse=False)
  yield_combiner.rewrite_module(query)
