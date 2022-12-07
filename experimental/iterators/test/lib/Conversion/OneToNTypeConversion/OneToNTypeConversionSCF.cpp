//===-- OneToNTypeConversionSCF.cpp - SCF 1:N type conversion ---*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The patterns in this file are heavily inspired (and copied from) upstream
// lib/Dialect/SCF/Transforms/StructuralTypeConversions.cpp but work for 1:N
// type conversions.
//
//===----------------------------------------------------------------------===//

#include "OneToNTypeConversionSCF.h"

#include "OneToNTypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace mlir::iterators;
using namespace mlir::scf;

class ConvertTypesInSCFIfOp : public OneToNOpConversionPattern<IfOp> {
public:
  using OneToNOpConversionPattern<IfOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>> matchAndRewrite(
      IfOp op, PatternRewriter &rewriter,
      const OneToNSignatureConversion & /*operandConversion*/,
      const OneToNSignatureConversion &resultConversion,
      const SmallVector<Value> & /*convertedOperands*/) const override {
    Location loc = op->getLoc();

    // Nothing to do if there is no non-identity conversion.
    if (!resultConversion.hasNonIdentityConversion())
      return failure();

    // Create new IfOp.
    ArrayRef<Type> convertedResultTypes = resultConversion.getConvertedTypes();
    auto newOp = rewriter.create<IfOp>(loc, convertedResultTypes,
                                       op.getCondition(), true);
    newOp->setAttrs(op->getAttrs());

    // We do not need the empty blocks created by rewriter.
    rewriter.eraseBlock(newOp.elseBlock());
    rewriter.eraseBlock(newOp.thenBlock());

    // Inlines block from the original operation.
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    return SmallVector<Value>(newOp->getResults());
  }
};

class ConvertTypesInSCFWhileOp : public OneToNOpConversionPattern<WhileOp> {
public:
  using OneToNOpConversionPattern<WhileOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(WhileOp op, PatternRewriter &rewriter,
                  const OneToNSignatureConversion &operandConversion,
                  const OneToNSignatureConversion &resultConversion,
                  const SmallVector<Value> &convertedOperands) const override {
    Location loc = op->getLoc();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!operandConversion.hasNonIdentityConversion() &&
        !resultConversion.hasNonIdentityConversion())
      return failure();

    // Create new WhileOp.
    ArrayRef<Type> convertedResultTypes = resultConversion.getConvertedTypes();

    auto newOp =
        rewriter.create<WhileOp>(loc, convertedResultTypes, convertedOperands);
    newOp->setAttrs(op->getAttrs());

    // Update block signatures.
    std::array<OneToNSignatureConversion, 2> blockConversions = {
        operandConversion, resultConversion};
    for (unsigned int i : {0u, 1u}) {
      Region *region = &op.getRegion(i);
      Block *block = &region->front();

      applySignatureConversion(block, blockConversions[i], rewriter);

      // Move updated region to new WhileOp.
      Region &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
    }

    return SmallVector<Value>(newOp->getResults());
  }
};

class ConvertTypesInSCFYieldOp : public OneToNOpConversionPattern<YieldOp> {
public:
  using OneToNOpConversionPattern<YieldOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(YieldOp op, PatternRewriter &rewriter,
                  const OneToNSignatureConversion &operandConversion,
                  const OneToNSignatureConversion & /*resultConversion*/,
                  const SmallVector<Value> &convertedOperands) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!operandConversion.hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.updateRootInPlace(op, [&] { op->setOperands(convertedOperands); });

    return SmallVector<Value>(op->getResults());
  }
};

class ConvertTypesInSCFConditionOp
    : public OneToNOpConversionPattern<ConditionOp> {
public:
  using OneToNOpConversionPattern<ConditionOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(ConditionOp op, PatternRewriter &rewriter,
                  const OneToNSignatureConversion &operandConversion,
                  const OneToNSignatureConversion & /*resultConversion*/,
                  const SmallVector<Value> &convertedOperands) const override {
    // Nothing to do if there is no non-identity conversion.
    if (!operandConversion.hasNonIdentityConversion())
      return failure();

    // Convert operands.
    rewriter.updateRootInPlace(op, [&] { op->setOperands(convertedOperands); });

    return SmallVector<Value>(op->getResults());
  }
};

namespace mlir {
namespace iterators {

void populateSCFTypeConversionPatterns(TypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertTypesInSCFConditionOp,
      ConvertTypesInSCFIfOp,
      ConvertTypesInSCFWhileOp,
      ConvertTypesInSCFYieldOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

} // namespace iterators
} // namespace mlir
