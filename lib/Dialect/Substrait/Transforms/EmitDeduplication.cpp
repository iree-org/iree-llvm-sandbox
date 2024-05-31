//===- EmitDeduplication.cpp - Impl. of emit deduplication ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Substrait/Transforms/Passes.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "structured/Dialect/Substrait/IR/Substrait.h"

namespace mlir::substrait {
#define GEN_PASS_DEF_SUBSTRAITEMITDEDUPLICATIONPASS
#include "structured/Dialect/Substrait/Transforms/Passes.h.inc"
} // namespace mlir::substrait

using namespace llvm;
using namespace mlir;
using namespace mlir::substrait;

namespace {

struct SubstraitEmitDeduplicationPass
    : substrait::impl::SubstraitEmitDeduplicationPassBase<
          SubstraitEmitDeduplicationPass> {
  using substrait::impl::SubstraitEmitDeduplicationPassBase<
      SubstraitEmitDeduplicationPass>::SubstraitEmitDeduplicationPassBase;

  void runOnOperation() override;
};

void SubstraitEmitDeduplicationPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  populateEmitDeduplicationPatterns(patterns);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns)))) {
    Location loc = getOperation()->getLoc();
    emitError(loc) << "emit deduplication: pattern application failed";
    signalPassFailure();
  }
}

/// If the given `input` was produced by an `emit` op with duplicates, creates a
/// new `emit` op without duplicates and returns the result of the new `emit`.
/// Otherwise, i.e., if the `input` was not produced by an `emit` op or that op
/// did not have duplicates, returns the original `input`. In both cases, also
/// populates `reverseMapping` with the mapping that re-establishes the original
/// order of the fields from the deduplicated order and returns the number of
/// fields after deduplication and whether the `input` was deduplicated.
std::tuple<Value, int64_t, bool>
createDeduplicatingEmit(Value input, SmallVector<int64_t> &reverseMapping,
                        PatternRewriter &rewriter) {
  // Handles the bases cases where the input either has no `emit` op or an
  // `emit` op with no duplicates. In that case, the returned value is just the
  // `input` and the reverse mapping is just the identity.
  auto handleNoDuplicates = [&]() {
    int64_t numInputFields = cast<TupleType>(input.getType()).getTypes().size();
    for (auto i : seq(numInputFields))
      reverseMapping.push_back(i);
    return std::tuple<Value, int64_t, bool>{input, numInputFields, false};
  };

  // Input is not an 'emit' op: handle base case.
  auto emitOp = llvm::dyn_cast_if_present<EmitOp>(input.getDefiningOp());
  if (!emitOp)
    return handleNoDuplicates();

  // Compute the new mapping without duplicates as well as, for each position in
  // the old mapping, the position in the new mapping.
  ArrayAttr oldInputMapping = emitOp.getMapping();
  SmallVector<int64_t> newInputMapping;
  SmallVector<int64_t> oldToNewInputMapping;
  {
    llvm::DenseMap<int64_t, int64_t> indexPositions;
    oldToNewInputMapping.reserve(oldInputMapping.size());
    for (auto [i, attr] : enumerate(oldInputMapping)) {
      int64_t index = cast<IntegerAttr>(attr).getInt();
      auto [it, success] = indexPositions.try_emplace(index, i);
      if (success)
        newInputMapping.push_back(index);
      oldToNewInputMapping.push_back(it->second);
    }
  }

  // If the new and old input mappings have the same size, then there are no
  // duplicates, so we handle it as a base case.
  if (newInputMapping.size() == oldInputMapping.size())
    return handleNoDuplicates();

  // Compute the mapping that re-establishes the original emit order.
  reverseMapping.reserve(reverseMapping.size() + newInputMapping.size());
  {
    // Compute the reverse mapping of the input.
    SmallVector<int64_t> reverseInputMapping(oldInputMapping.size());
    for (auto [i, index] : enumerate(newInputMapping))
      reverseInputMapping[index] = i;

    // The first fields of the reverse mapping reverse the effect of the
    // deduplication of the emit op on the input.
    for (auto [i, attr] :
         enumerate(oldInputMapping.getAsRange<IntegerAttr>())) {
      int64_t reverseIndex = reverseInputMapping[attr.getInt()];
      reverseMapping.push_back(reverseIndex);
    }
  }

  // If we did have duplicates, add an `emit` op that deduplicates the input.
  Location loc = emitOp.getLoc();
  ArrayAttr newInputMappingAttr = rewriter.getI64ArrayAttr(newInputMapping);
  auto newEmitOp =
      rewriter.create<EmitOp>(loc, emitOp.getInput(), newInputMappingAttr);

  return {newEmitOp, newInputMapping.size(), true};
}

/// Pushes duplicates in the mappings of `emit` ops producing either of the two
/// inputs through the `cross` op. This works by introducing new emit ops
/// without the duplicates, creating a new `cross` op that uses them, and
/// finally a new `emit` op that maps back to the original order.
struct PushDuplicateThroughCrossJoinPattern : public OpRewritePattern<CrossOp> {
  using OpRewritePattern<CrossOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CrossOp op,
                                PatternRewriter &rewriter) const override {
    bool isLeftEmit = isa_and_present<EmitOp>(op.getLeft().getDefiningOp());
    bool isRightEmit = isa_and_present<EmitOp>(op.getRight().getDefiningOp());
    if (!isLeftEmit && !isRightEmit)
      return rewriter.notifyMatchFailure(
          op, "none of the operands is an 'emit' op");

    // Create input ops for the new `cross` op. These may be the original inputs
    // or `emit` ops that remove duplicates.

    // Left input: the reverse mapping of the left input works as the prefix of
    // the reverse mapping of the new `cross` op.
    SmallVector<int64_t> reverseMapping;
    auto [newLeftInput, numLeftIndices, leftHasDuplicates] =
        createDeduplicatingEmit(op.getLeft(), reverseMapping, rewriter);

    // Right input: the reverse mapping of the right input needs to be adjusted
    // by the number of deduplicated fields in the left input.
    int64_t numLeftOriginalindices = reverseMapping.size();
    auto [newRightInput, numRightIndices, rightHasDuplicates] =
        createDeduplicatingEmit(op.getRight(), reverseMapping, rewriter);
    for (int64_t &idx : drop_begin(reverseMapping, numLeftOriginalindices))
      idx += numLeftIndices;

    if (!leftHasDuplicates && !rightHasDuplicates)
      // Note: if we end up failing here, then both invocations of
      // `createDeduplicatingEmit` returned without creating a new (`emit`) op.
      return rewriter.notifyMatchFailure(
          op, "none of the 'emit' inputs have duplicates");

    // Create new cross op with the two deduplicated inputs.
    auto newOp =
        rewriter.create<CrossOp>(op.getLoc(), newLeftInput, newRightInput);

    // Replace old cross op with emit op that maps back to old emit order.
    ArrayAttr reverseMappingAttr = rewriter.getI64ArrayAttr(reverseMapping);
    rewriter.replaceOpWithNewOp<EmitOp>(op, newOp, reverseMappingAttr);

    return success();
  }
};

} // namespace

namespace mlir {
namespace substrait {

void populateEmitDeduplicationPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<PushDuplicateThroughCrossJoinPattern>(context);
}

std::unique_ptr<Pass> createEmitDeduplicationPass() {
  return std::make_unique<SubstraitEmitDeduplicationPass>();
}

} // namespace substrait
} // namespace mlir
