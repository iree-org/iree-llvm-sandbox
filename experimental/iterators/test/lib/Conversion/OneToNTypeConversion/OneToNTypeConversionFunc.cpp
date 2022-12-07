//===-- OneToNTypeConversionFunc.cpp - Func 1:N type conversion -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The patterns in this file are heavily inspired (and copied from) upstream
// convertFuncOpTypes in lib/Transforms/Utils/DialectConversion.cpp and the
// patterns in lib/Dialect/Func/Transforms/FuncConversions.cpp but work for 1:N
// type conversions.
//
//===----------------------------------------------------------------------===//

#include "OneToNTypeConversionFunc.h"

#include "OneToNTypeConversion.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;
using namespace mlir::func;
using namespace mlir::iterators;

class ConvertTypesInFuncCallOp : public OneToNOpConversionPattern<CallOp> {
public:
  using OneToNOpConversionPattern<CallOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(CallOp op, PatternRewriter &rewriter,
                  const OneToNSignatureConversion &operandConversion,
                  const OneToNSignatureConversion &resultConversion,
                  const SmallVector<Value> &convertedOperands) const override {
    Location loc = op->getLoc();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!operandConversion.hasNonIdentityConversion() &&
        !resultConversion.hasNonIdentityConversion())
      return failure();

    // Create new CallOp.
    auto newOp = rewriter.create<CallOp>(
        loc, resultConversion.getConvertedTypes(), convertedOperands);
    newOp->setAttrs(op->getAttrs());

    return SmallVector<Value>(newOp->getResults());
  }
};

class ConvertTypesInFuncFuncOp : public OneToNOpConversionPattern<FuncOp> {
public:
  using OneToNOpConversionPattern<FuncOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>> matchAndRewrite(
      FuncOp op, PatternRewriter &rewriter,
      const OneToNSignatureConversion & /*operandConversion*/,
      const OneToNSignatureConversion & /*resultConversion*/,
      const SmallVector<Value> & /*convertedOperands*/) const override {
    // Construct conversion mapping for arguments.
    OneToNSignatureConversion argumentConversion(op.getArgumentTypes());
    if (failed(typeConverter->convertSignatureArgs(op.getArgumentTypes(),
                                                   argumentConversion)))
      return failure();

    // Construct conversion mapping for arguments.
    OneToNSignatureConversion funcResultConversion(op.getResultTypes());
    if (failed(typeConverter->convertSignatureArgs(op.getResultTypes(),
                                                   funcResultConversion)))
      return failure();

    // Nothing to do if the op doesn't have any non-identity conversions for its
    // operands or results.
    if (!argumentConversion.hasNonIdentityConversion() &&
        !funcResultConversion.hasNonIdentityConversion())
      return failure();

    // Update the function signature in-place.
    auto newType = FunctionType::get(rewriter.getContext(),
                                     argumentConversion.getConvertedTypes(),
                                     funcResultConversion.getConvertedTypes());
    rewriter.updateRootInPlace(op, [&] { op.setType(newType); });

    // Update block signatures.
    if (!op.isExternal()) {
      Region *region = &op.getBody();
      Block *block = &region->front();
      applySignatureConversion(block, argumentConversion, rewriter);
    }

    return SmallVector<Value>(op->getResults());
  }
};

class ConvertTypesInFuncReturnOp : public OneToNOpConversionPattern<ReturnOp> {
public:
  using OneToNOpConversionPattern<ReturnOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(ReturnOp op, PatternRewriter &rewriter,
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

void populateFuncTypeConversionPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertTypesInFuncCallOp,
      ConvertTypesInFuncFuncOp,
      ConvertTypesInFuncReturnOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

} // namespace iterators
} // namespace mlir
