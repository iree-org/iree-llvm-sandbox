//===-- OneToNTypeConversion.cpp - Utils for 1:N type conversion-*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OneToNTypeConversion.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::iterators;

std::optional<SmallVector<Value>>
OneToNTypeConverter::materializeTargetConversion(OpBuilder &builder,
                                                 Location loc,
                                                 TypeRange resultTypes,
                                                 Value input) const {
  for (const OneToNMaterializationCallbackFn &fn :
       llvm::reverse(oneToNTargetMaterializations)) {
    if (std::optional<SmallVector<Value>> result =
            fn(builder, resultTypes, input, loc))
      return *result;
  }
  return {};
}

TypeRange OneToNTypeMapping::getConvertedTypes(unsigned originalTypeNo) const {
  TypeRange convertedTypes = getConvertedTypes();
  if (auto mapping = getInputMapping(originalTypeNo))
    return convertedTypes.slice(mapping->inputNo, mapping->size);
  return {};
}

ValueRange
OneToNTypeMapping::getConvertedValues(ValueRange convertedValues,
                                      unsigned originalValueNo) const {
  if (auto mapping = getInputMapping(originalValueNo))
    return convertedValues.slice(mapping->inputNo, mapping->size);
  return {};
}

void OneToNTypeMapping::convertLocation(
    Value originalValue, unsigned originalValueNo,
    llvm::SmallVectorImpl<Location> &result) const {
  if (auto mapping = getInputMapping(originalValueNo))
    result.append(mapping->size, originalValue.getLoc());
}

void OneToNTypeMapping::convertLocations(
    ValueRange originalValues, llvm::SmallVectorImpl<Location> &result) const {
  assert(originalValues.size() == getOriginalTypes().size());
  for (auto &[i, value] : llvm::enumerate(originalValues))
    convertLocation(value, i, result);
}

static bool isIdentityConversion(Type originalType, TypeRange convertedTypes) {
  return convertedTypes.size() == 1 && convertedTypes[0] == originalType;
}

bool OneToNTypeMapping::hasNonIdentityConversion() const {
  // XXX: I think that the original types and the converted types are the same
  //      iff there was no non-identity type conversion. If that is true, the
  //      patterns could actually test whether there is anything useful to do
  //      without having access to the signature conversion.
  for (auto [i, originalType] : llvm::enumerate(originalTypes)) {
    TypeRange types = getConvertedTypes(i);
    if (!isIdentityConversion(originalType, types)) {
      assert(TypeRange(originalTypes) != getConvertedTypes());
      return true;
    }
  }
  assert(TypeRange(originalTypes) == getConvertedTypes());
  return false;
}

/// Builds an UnrealizedConversionCastOp from the given inputs to the given
/// result types. Returns the result values of the cast.
static ValueRange buildUnrealizedCast(OpBuilder &builder, TypeRange resultTypes,
                                      ValueRange inputs) {
  Location loc = builder.getUnknownLoc();
  if (!inputs.empty())
    loc = inputs.front().getLoc();
  auto castOp =
      builder.create<UnrealizedConversionCastOp>(loc, resultTypes, inputs);
  return castOp->getResults();
}

/// Builds one UnrealizedConversionCastOp for each of the given original values
/// using the respective target types given in the provided conversion mapping
/// and returns the results of these casts. If the conversion mapping of a value
/// maps a type to itself (i.e., is an identity conversion), then no cast is
/// inserted and the original value is returned instead.
/// Note that these unrealized are different from target materializations in
/// that they are *always* inserted, even if they immediately fold away, such
/// that patterns always see valid intermediate IR, whereas materilizations are
/// only used in the places where the unrealized casts *don't* fold away.
static SmallVector<Value>
buildUnrealizedForwardCasts(ValueRange originalValues,
                            OneToNTypeMapping &conversion,
                            RewriterBase &rewriter) {

  // Convert each operand one by one.
  SmallVector<Value> convertedValues;
  convertedValues.reserve(conversion.getConvertedTypes().size());
  for (auto [idx, originalValue] : llvm::enumerate(originalValues)) {
    TypeRange convertedTypes = conversion.getConvertedTypes(idx);

    // Identity conversion: keep operand as is.
    if (isIdentityConversion(originalValue.getType(), convertedTypes)) {
      convertedValues.push_back(originalValue);
      continue;
    }

    // Non-identity conversion: materialize target types.
    ValueRange castResult =
        buildUnrealizedCast(rewriter, convertedTypes, originalValue);
    convertedValues.append(castResult.begin(), castResult.end());
  }

  return convertedValues;
}

/// Builds one UnrealizedConversionCastOp for each sequence of the given
/// original values to one value of the type they originated from, i.e., a
/// "reverse" conversion from N converted values back to one value of the
/// original type, using the given (forward) type conversion. If a given value
/// was mapped to a value of the same type (i.e., the conversion in the mapping
/// is an identity conversion), then the "converted" value is returned without
/// cast.
/// Note that these unrealized are different from source materializations in
/// that they are *always* inserted, even if they immediately fold away, such
/// that patterns always see valid intermediate IR, whereas materilizations are
/// only used in the places where the unrealized casts *don't* fold away.
static SmallVector<Value>
buildUnrealizedBackwardsCasts(ValueRange convertedValues,
                              const OneToNTypeMapping &typeConversion,
                              RewriterBase &rewriter) {
  assert(typeConversion.getConvertedTypes() == convertedValues.getTypes());

  // Create unrealized cast op for each converted result of the op.
  SmallVector<Value> recastValues;
  TypeRange originalTypes = typeConversion.getOriginalTypes();
  recastValues.reserve(originalTypes.size());
  auto convertedValueIt = convertedValues.begin();
  for (auto [idx, originalType] : llvm::enumerate(originalTypes)) {
    TypeRange convertedTypes = typeConversion.getConvertedTypes(idx);
    size_t numConvertedValues = convertedTypes.size();
    if (isIdentityConversion(originalType, convertedTypes)) {
      // Identity conversion: take result as is.
      recastValues.push_back(*convertedValueIt);
    } else {
      // Non-identity conversion: cast back to source type.
      ValueRange recastValue = buildUnrealizedCast(
          rewriter, originalType,
          ValueRange{convertedValueIt, convertedValueIt + numConvertedValues});
      assert(recastValue.size() == 1);
      recastValues.push_back(recastValue.front());
    }
    convertedValueIt += numConvertedValues;
  }

  return recastValues;
}

LogicalResult
OneToNConversionPattern::matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter) const {
  auto *typeConverter = getTypeConverter<OneToNTypeConverter>();

  // Construct conversion mapping for results.
  Operation::result_type_range originalResultTypes = op->getResultTypes();
  OneToNTypeMapping resultMapping(originalResultTypes);
  if (failed(typeConverter->computeTypeMapping(originalResultTypes,
                                               resultMapping)))
    return failure();

  // Construct conversion mapping for operands.
  Operation::operand_type_range originalOperandTypes = op->getOperandTypes();
  OneToNTypeMapping operandMapping(originalOperandTypes);
  if (failed(typeConverter->computeTypeMapping(originalOperandTypes,
                                               operandMapping)))
    return failure();

  // Cast operands to target types.
  SmallVector<Value> convertedOperands =
      buildUnrealizedForwardCasts(op->getOperands(), operandMapping, rewriter);

  // Apply actual pattern.
  auto result = matchAndRewrite(op, rewriter, operandMapping, resultMapping,
                                convertedOperands);

  if (failed(result))
    return failure();
  SmallVector<Value> &replacementValues = result.value();

  // If replacementValues consist of the results of the original op, assume
  // in-place update.
  // TODO: This isn't particularly elegant. Not sure how else to handle that
  //       case without tracking modifications through the rewriter, which
  //       would require a custom pattern application driver.
  if (ValueRange{op->getResults()} == replacementValues)
    return success();

  // Cast op results back to the original types and use those.
  SmallVector<Value> castResults =
      buildUnrealizedBackwardsCasts(replacementValues, resultMapping, rewriter);
  rewriter.replaceOp(op, castResults);

  return success();
}

namespace mlir {
namespace iterators {
Block *applySignatureConversion(Block *block,
                                OneToNTypeMapping &argumentConversion,
                                RewriterBase &rewriter) {
  // Split the block at the beginning to get a new block to use for the
  // updated signature.
  SmallVector<Location> locs;
  argumentConversion.convertLocations(block->getArguments(), locs);
  Block *newBlock =
      rewriter.createBlock(block, argumentConversion.getConvertedTypes(), locs);
  rewriter.replaceAllUsesWith(block, newBlock);

  // Create necessary casts in new block.
  SmallVector<Value> castResults;
  for (auto [i, arg] : llvm::enumerate(block->getArguments())) {
    TypeRange convertedTypes = argumentConversion.getConvertedTypes(i);
    ValueRange newArgs =
        argumentConversion.getConvertedValues(newBlock->getArguments(), i);
    if (isIdentityConversion(arg.getType(), convertedTypes)) {
      // Identity conversion: take argument as is.
      assert(newArgs.size() == 1);
      castResults.push_back(newArgs.front());
    } else {
      // Non-identity conversion: cast the converted arguments to the original
      // type.
      PatternRewriter::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(newBlock);
      ValueRange castResult =
          buildUnrealizedCast(rewriter, arg.getType(), newArgs);
      assert(castResult.size() == 1);
      castResults.push_back(castResult.front());
    }
  }

  // Merge old block into new block such that we only have the latter with the
  // new signature.
  rewriter.mergeBlocks(block, newBlock, castResults);

  return newBlock;
}

// This function applies the provided patterns using
// applyPatternsAndFoldGreedily and then replaces all newly inserted
// UnrealizedConversionCastOps that haven't folded away. ("Backward" casts from
// target to source types inserted by a OneToNConversionPattern normally fold
// away with the "forward" casts from source to target types inserted by the
// next pattern.) To understand which casts are "newly inserted", we save a list
// of all casts existing before the patterns are applied and assume that all
// casts not in that list after the application are new. (This is probably not
// correct: It might be possible that an existing cast is folded away and a new
// cast happens to be allocated with exactly the same pointer. Dealing with that
// possiblity is an open TODO.) Also, we do not track which inserted casts are
// needed for source, target, or argument materialization, so we do some
// educated guessing to recover that information. Fixing both issues would
// require to use a PatternRewriter that overloads various `notify*` functions
// and similar and tracks all changes there. However, that would require a
// dedicated pattern application driver, which is currently also left as an open
// TODO.)
LogicalResult applyOneToNConversion(Operation *op,
                                    OneToNTypeConverter &typeConverter,
                                    const FrozenRewritePatternSet &patterns) {
  // Remember existing unrealized casts.
  SmallSet<UnrealizedConversionCastOp, 4> existingCasts;
  op->walk(
      [&](UnrealizedConversionCastOp castOp) { existingCasts.insert(castOp); });

  // Apply provided conversion patterns.
  if (failed(applyPatternsAndFoldGreedily(op, patterns)))
    return failure();

  // Find all newly inserted unrealized casts (that haven't folded away).
  SmallVector<UnrealizedConversionCastOp> worklist;
  op->walk([&](UnrealizedConversionCastOp castOp) {
    if (!existingCasts.contains(castOp))
      worklist.push_back(castOp);
  });

  // Replace new casts with user materializations.
  IRRewriter rewriter(op->getContext());
  for (UnrealizedConversionCastOp castOp : worklist) {
    // Create user materialization.
    TypeRange resultTypes = castOp->getResultTypes();
    rewriter.setInsertionPoint(castOp);
    SmallVector<Value> materializedResults;

    // Determine whether operands or results are already legal to know which
    // kind of materilization this is.
    ValueRange operands = castOp.getOperands();
    bool areOperandTypesLegal = llvm::all_of(
        operands.getTypes(), [&](Type t) { return typeConverter.isLegal(t); });
    bool areResultsTypesLegal = llvm::all_of(
        resultTypes, [&](Type t) { return typeConverter.isLegal(t); });

    if (!areOperandTypesLegal && areResultsTypesLegal && operands.size() == 1) {
      // This is a target materilization.
      std::optional<SmallVector<Value>> maybeResults =
          typeConverter.materializeTargetConversion(
              rewriter, castOp->getLoc(), resultTypes, operands.front());
      if (!maybeResults)
        return failure();
      materializedResults = maybeResults.value();
    } else if (areOperandTypesLegal && !areResultsTypesLegal &&
               resultTypes.size() == 1) {
      // This is a source or an argument materialization.
      std::optional<Value> maybeResult;
      if (llvm::all_of(operands, [&](Value v) { return v.getDefiningOp(); })) {
        // This is an source materialization.
        maybeResult = typeConverter.materializeArgumentConversion(
            rewriter, castOp->getLoc(), resultTypes.front(),
            castOp.getOperands());
      } else {
        // This is an argument materialization.
        maybeResult = typeConverter.materializeSourceConversion(
            rewriter, castOp->getLoc(), resultTypes.front(),
            castOp.getOperands());
      }
      if (!maybeResult.has_value() || !maybeResult.value())
        return failure();
      materializedResults = {maybeResult.value()};
    } else {
      assert(false && "unexpected cast inserted");
    }

    // Replace cast with materialization.
    rewriter.replaceOp(castOp, materializedResults);
  }

  return success();
}
} // namespace iterators
} // namespace mlir
