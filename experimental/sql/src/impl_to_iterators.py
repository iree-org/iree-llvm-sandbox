# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from xdsl.ir import Operation, MLContext, Region, Block, Attribute
from typing import List, Type, Optional
from xdsl.dialects.builtin import ArrayAttr, StringAttr, ModuleOp, IntegerAttr, IntegerType, TupleType
from xdsl.dialects.llvm import LLVMStructType, LLVMExtractValue, LLVMInsertValue
from xdsl.dialects.func import FuncOp, Return
from xdsl.dialects.arith import Addi
from xdsl.dialects.func import Return

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
    if isinstance(type_, RelImpl.Int64):
      # TODO: this is obviously broken, but the backend currently only support i32
      return IntegerType.from_width(32)
    raise Exception(f"type conversion not yet implemented for {type(type_)}")

  def convert_bag(self, bag: RelImpl.Bag) -> it.Stream:
    types = [self.convert_datatype(s.elt_type) for s in bag.schema.data]
    return it.Stream.get(LLVMStructType.from_type_list(types))

  def add_sum_function(self, region: Region, elem_types: List[Attribute]):
    struct_type = LLVMStructType.from_type_list(elem_types)
    index_attr = IntegerAttr.from_index_int_value(0)
    sum_struct = FuncOp.from_region(
        "sum_struct",
        [struct_type, struct_type],
        [struct_type],
        Region.from_block_list([
            Block.from_callable(
                [struct_type, struct_type],
                lambda ba1, ba2: [
                    l := LLVMExtractValue.
                    build(result_types=elem_types,
                          attributes=
                          {"position": ArrayAttr.from_list([index_attr])},
                          operands=[ba1]),
                    # This comment is needed to save formatting
                    r := LLVMExtractValue.build(
                        result_types=elem_types,
                        attributes=
                        {"position": ArrayAttr.from_list([index_attr])},
                        operands=[ba2]),
                    s := Addi.get(l, r),
                    res := LLVMInsertValue.build(
                        result_types=[struct_type],
                        operands=[ba1, s],
                        attributes=
                        {"position": ArrayAttr.from_list([index_attr])}),
                    Return.get(res)
                ])
        ]))
    region.blocks[0].add_op(sum_struct)
    return


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#

#===------------------------------------------------------------------------===#
# Operators
#===------------------------------------------------------------------------===#


@dataclass
class FullTableScanRewriter(RelImplRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelImpl.FullTableScanOp,
                        rewriter: PatternRewriter):
    # TODO: Change this once loading from input is supported
    rewriter.replace_matched_op(
        it.ConstantStreamOp.get([[IntegerAttr.from_int_and_width(0, 32)],
                                 [IntegerAttr.from_int_and_width(1, 32)],
                                 [IntegerAttr.from_int_and_width(2, 32)],
                                 [IntegerAttr.from_int_and_width(3, 32)]],
                                self.convert_bag(op.result.typ)))


@dataclass
class AggregateRewriter(RelImplRewriter):

  @op_type_rewrite_pattern
  def match_and_rewrite(self, op: RelImpl.Aggregate, rewriter: PatternRewriter):
    self.add_sum_function(op.parent_op().parent_region(),
                          self.convert_bag(op.result.typ).types.types.data)
    rewriter.replace_matched_op(
        it.ReduceOp.get(op.input.op, StringAttr.from_str("sum_struct"),
                        self.convert_bag(op.result.typ)))


#===------------------------------------------------------------------------===#
# Conversion setup
#===------------------------------------------------------------------------===#


def impl_to_iterators(ctx: MLContext, query: ModuleOp):

  f = FuncOp.from_region("main", [], [],
                         Region.from_block_list([query.body.detach_block(0)]))
  query.body.add_block(Block.from_ops([f]))
  walker = PatternRewriteWalker(GreedyRewritePatternApplier(
      [FullTableScanRewriter(), AggregateRewriter()]),
                                walk_regions_first=False,
                                apply_recursively=False,
                                walk_reverse=False)
  walker.rewrite_module(query)
  # Adding the sink
  query.body.blocks[0].ops[0].body.blocks[0].add_op(
      it.SinkOp.get(query.body.blocks[0].ops[0].body.blocks[0].ops[-1]))
  # Adding the return
  query.body.blocks[0].ops[0].body.blocks[0].add_op(Return.get())
  # Wrapping everything into a main function
