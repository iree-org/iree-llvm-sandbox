//===- EmitDeduplication.cpp - Impl. of emit deduplication ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Substrait/Transforms/Passes.h"

#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/CSE.h"
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

/// Deduplicates the fields of the region with a single `Tuple` argument using
/// the provided (deduplicating) mapping. This involves changing the type of the
/// region argument to the provided `newElementType`, which must be the type
/// obtained by applying deduplication to the argument type of the provided
/// `region`, as well as changing all `field_reference` ops using the region
/// argument to work on the deduplicated type.
// TODO(ingomueller): We could add an overload for this function that computes
// `newElementType` from the type of the region argument and the mapping.
void deduplicateRegionArgs(Region &region, ArrayAttr newMapping,
                           Type newElementType, PatternRewriter &rewriter) {
  assert(region.getNumArguments() == 1 &&
         "only regions with 1 argument are supported");
  auto oldElementType = cast<TupleType>(region.getArgument(0).getType());
  int64_t numOldFields = oldElementType.getTypes().size();

  // For each position in the original input type, compute which position it
  // corresponds to in the deduplicated input. This is required for replacing
  // field references to the original type with references to the deduplicated
  // type.
  SmallVector<int64_t> oldToNewPositions;
  oldToNewPositions.reserve(numOldFields);
  {
    llvm::DenseMap<int64_t, int64_t> indexPositions;
    for (auto attr : newMapping.getAsRange<IntegerAttr>()) {
      int64_t index = attr.getInt();
      int64_t pos = indexPositions.size();
      auto [it, success] = indexPositions.try_emplace(index, pos);
      oldToNewPositions.push_back(it->second);
    }
  }

  // Update field references using the region argument.
  for (Operation *user : region.getArgument(0).getUsers()) {
    // We are only interested in `field_reference` ops.
    if (!isa<FieldReferenceOp>(user))
      continue;
    auto refOp = cast<FieldReferenceOp>(user);

    // Compute new position array from the old one.
    ArrayRef<int64_t> oldPositions = refOp.getPosition();
    SmallVector<int64_t> newPositions;
    newPositions.reserve(oldPositions.size());
    for (auto index : oldPositions)
      newPositions.push_back(index);
    newPositions[0] = oldToNewPositions[newPositions[0]];

    // Update op in place.
    refOp.setPosition(newPositions);
  }

  // Update argument type of the region.
  region.getArgument(0).setType(newElementType);
}

struct EliminateDuplicateYieldsInProjectPattern
    : public OpRewritePattern<ProjectOp> {
  using OpRewritePattern<ProjectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ProjectOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *context = op.getContext();
    Operation *terminator = op.getExpressions().front().getTerminator();
    int64_t numOriginalYields = terminator->getNumOperands();
    auto inputTupleType = cast<TupleType>(op.getInput().getType());

    // Determine duplicate values in `yield` and remember the first ocurrence of
    // each value.
    llvm::DenseMap<Value, int64_t> valuePositions;
    for (Value value : terminator->getOperands())
      valuePositions.try_emplace(value, valuePositions.size());

    if (valuePositions.size() == numOriginalYields)
      return rewriter.notifyMatchFailure(op, "does not yield duplicate values");

    // Create a mapping from the de-duplicated values that re-establishes the
    // original emit order. The input fields are just forwarded, so create
    // identity prefix.
    SmallVector<int64_t> reverseMapping;
    reverseMapping.reserve(inputTupleType.size() + numOriginalYields);
    append_range(reverseMapping, iota_range<int64_t>(0, inputTupleType.size(),
                                                     /*Inclusive=*/false));

    // Reverse mapping: The fields added by the `expression` regions are now
    // de-duplicated, so we need to reverse the effect of the deduplication,
    // taking the prefix into account.
    for (Value value : terminator->getOperands()) {
      int64_t pos = valuePositions[value];
      reverseMapping.push_back(inputTupleType.size() + pos);
    }

    // Remove duplicate values in `yield` op of the `expressions` region.
    {
      SmallVector<Value> values;
      values.reserve(valuePositions.size());
      for (auto [value, pos] : valuePositions)
        values.push_back(value);

      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(terminator);
      terminator = rewriter.replaceOpWithNewOp<YieldOp>(terminator, values);
    }

    // Compute deduplicated output field types.
    SmallVector<Type> outputTypes;
    int64_t numNewYields = terminator->getNumOperands();
    outputTypes.reserve(inputTupleType.size() + numNewYields);
    append_range(outputTypes, inputTupleType.getTypes());
    append_range(outputTypes, terminator->getOperandTypes());
    auto newOutputType = TupleType::get(context, outputTypes);

    // Create new `project` op with updated region and output type.
    auto newOp =
        rewriter.create<ProjectOp>(op.getLoc(), newOutputType, op.getInput());
    rewriter.inlineRegionBefore(op.getExpressions(), newOp.getExpressions(),
                                newOp.getExpressions().end());

    // Create `emit` op with the reverse mapping.
    ArrayAttr reverseMappingAttr = rewriter.getI64ArrayAttr(reverseMapping);
    rewriter.replaceOpWithNewOp<EmitOp>(op, newOp, reverseMappingAttr);

    return success();
  }
};

/// Pushes duplicates in the mappings of `emit` ops producing either of the two
/// inputs through the `cross` op. This works by introducing new emit ops
/// without the duplicates, creating a new `cross` op that uses them, and
/// finally a new `emit` op that maps back to the original order.
struct PushDuplicatesThroughCrossPattern : public OpRewritePattern<CrossOp> {
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

/// Pushes duplicates in the mappings of `emit` ops producing the input through
/// the `filter` op. This works by introducing a new `emit` op without the
/// duplicates, creating a new `filter` op updated to work on the deduplicated
/// element type, and finally a new `emit` op that maps back to the original
/// order.
struct PushDuplicatesThroughFilterPattern : public OpRewritePattern<FilterOp> {
  using OpRewritePattern<FilterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(FilterOp op,
                                PatternRewriter &rewriter) const override {
    auto emitOp = op.getInput().getDefiningOp<EmitOp>();
    if (!emitOp)
      return rewriter.notifyMatchFailure(
          op, "input operand is not produced by an 'emit' op");

    // Create input ops for the new `filter` op. These may be the original
    // inputs or `emit` ops that remove duplicates.
    SmallVector<int64_t> reverseMapping;
    auto [newInput, numDedupIndices, hasDuplicates] =
        createDeduplicatingEmit(op.getInput(), reverseMapping, rewriter);

    if (!hasDuplicates)
      // Note: if we end up failing here, then the invokation of
      // `createDeduplicatingEmit` returned without creating a new (`emit`) op.
      return rewriter.notifyMatchFailure(
          op, "the 'emit' input does not have duplicates");

    // Create new `filter` op. Move over the `condition` region. This needs to
    // happen now because replacing the op will destroy the region.
    auto newOp = rewriter.create<FilterOp>(op.getLoc(), newInput);
    rewriter.inlineRegionBefore(op.getCondition(), newOp.getCondition(),
                                newOp.getCondition().end());

    // Update the `condition` region.
    deduplicateRegionArgs(newOp.getCondition(), emitOp.getMapping(),
                          newInput.getType(), rewriter);

    // Deduplicating block args may create common subexpressions. Eliminate
    // them immediately.
    {
      DominanceInfo domInfo;
      mlir::eliminateCommonSubExpressions(rewriter, domInfo, newOp);
    }

    // Replace the old `filter` op with a new `emit` op that maps back to the
    // original emit order.
    ArrayAttr reverseMappingAttr = rewriter.getI64ArrayAttr(reverseMapping);
    rewriter.replaceOpWithNewOp<EmitOp>(op, newOp, reverseMappingAttr);

    return failure();
  }
};

/// Pushes duplicates in the mappings of `emit` ops producing the input through
/// the `filter` op. This works by introducing a new `emit` op without the
/// duplicates, creating a new `filter` op updated to work on the deduplicated
/// element type, and finally a new `emit` op that maps back to the original
/// order.
struct PushDuplicatesThroughProjectPattern
    : public OpRewritePattern<ProjectOp> {
  using OpRewritePattern<ProjectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ProjectOp op,
                                PatternRewriter &rewriter) const override {
    auto emitOp = op.getInput().getDefiningOp<EmitOp>();
    if (!emitOp)
      return rewriter.notifyMatchFailure(
          op, "input operand is not produced by an 'emit' op");

    // Create input ops for the new `project` op. These may be the original
    // inputs or `emit` ops that remove duplicates.
    SmallVector<int64_t> reverseMapping;
    auto [newInput, numDedupIndices, hasDuplicates] =
        createDeduplicatingEmit(op.getInput(), reverseMapping, rewriter);

    if (!hasDuplicates)
      // Note: if we end up failing here, then the invokation of
      // `createDeduplicatingEmit` returned without creating a new (`emit`) op.
      return rewriter.notifyMatchFailure(
          op, "the 'emit' input does not have duplicates");

    MLIRContext *context = op.getContext();

    // Compute deduplicated output field types.
    Operation *terminator = op.getExpressions().front().getTerminator();
    auto newInputTupleType = cast<TupleType>(newInput.getType());

    SmallVector<Type> outputTypes;
    outputTypes.reserve(newInputTupleType.size() +
                        terminator->getNumOperands());
    append_range(outputTypes, newInputTupleType.getTypes());
    append_range(outputTypes, terminator->getOperandTypes());
    auto newOutputType = TupleType::get(context, outputTypes);

    // Create new `project` op. Move over the `expressions` region. This needs
    // to happen now because replacing the op will destroy the region.
    auto newOp =
        rewriter.create<ProjectOp>(op.getLoc(), newOutputType, newInput);
    rewriter.inlineRegionBefore(op.getExpressions(), newOp.getExpressions(),
                                newOp.getExpressions().end());

    // Update the `condition` region.
    deduplicateRegionArgs(newOp.getExpressions(), emitOp.getMapping(),
                          newInput.getType(), rewriter);

    // Deduplicating block args may create common subexpressions. Eliminate
    // them immediately.
    {
      DominanceInfo domInfo;
      mlir::eliminateCommonSubExpressions(rewriter, domInfo, newOp);
    }

    // Compute output indices for the expressions added by the region.
    int64_t numTotalIndices = numDedupIndices + terminator->getNumOperands();
    append_range(reverseMapping, seq(numDedupIndices, numTotalIndices));

    // Replace the old `project` op with a new `emit` op that maps back to the
    // original emit order.
    ArrayAttr reverseMappingAttr = rewriter.getI64ArrayAttr(reverseMapping);
    rewriter.replaceOpWithNewOp<EmitOp>(op, newOp, reverseMappingAttr);

    return failure();
  }
};

} // namespace

namespace mlir {
namespace substrait {

void populateEmitDeduplicationPatterns(RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<
      // clang-format off
      EliminateDuplicateYieldsInProjectPattern,
      PushDuplicatesThroughCrossPattern,
      PushDuplicatesThroughFilterPattern,
      PushDuplicatesThroughProjectPattern
      // clang-format on
      >(context);
}

std::unique_ptr<Pass> createEmitDeduplicationPass() {
  return std::make_unique<SubstraitEmitDeduplicationPass>();
}

} // namespace substrait
} // namespace mlir
