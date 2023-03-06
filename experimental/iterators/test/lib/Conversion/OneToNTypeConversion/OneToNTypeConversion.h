//===-- OneToNTypeConversion.h - Utils for 1:N type conversion --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides utils for implementing (poor-man's) dialect conversion
// passes with 1:N type conversions.
//
// The main function first applies a set of RewritePatterns, which produce
// unrealized casts to convert the operands and results from and to the source
// types, and then replaces all newly added unrealized casts by user-provided
// materializations. For this to work, the main function requires a special
// TypeConverter and special RewritePatterns, respectively deriving from the
// provided classes, which extend their respective base classes for 1:N type
// converions.
//
// Note that this is much more simple-minded than the "real" dialect conversion,
// which checks for legality before applying patterns and does probably many
// other additional things. Ideally, some of the extensions here could be
// integrated there.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_LIB_CONVERSION_ONETONTYPECONVERSION_ONETONTYPECONVERSION_H
#define TEST_LIB_CONVERSION_ONETONTYPECONVERSION_ONETONTYPECONVERSION_H

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace iterators {

/// Extends `TypeConverter` with 1:N target materializations. Such
/// materializations have to provide the "reverse" of 1:N type conversions,
/// i.e., they need to materialize N values with target types into one value
/// with a source type (which isn't possible in the base class currently).
class OneToNTypeConverter : public TypeConverter {
public:
  using OneToNMaterializationCallbackFn =
      std::function<std::optional<SmallVector<Value>>(OpBuilder &, TypeRange,
                                                      Value, Location)>;

  /// Creates the mapping of the given range of original types to target types
  /// of the conversion and stores that mapping in the given (signature)
  /// conversion. This function simply calls TypeConverter::convertSignatureArgs
  /// and exists here with a different name to reflect the broader semantic.
  LogicalResult computeTypeMapping(TypeRange types,
                                   SignatureConversion &result) {
    return convertSignatureArgs(types, result);
  }

  /// Applies one of the user-provided 1:N target materializations (in LIFO
  /// order).
  std::optional<SmallVector<Value>>
  materializeTargetConversion(OpBuilder &builder, Location loc,
                              TypeRange resultTypes, Value input) const;

  /// Adds a 1:N target materialization to the converter. Such materializations
  /// build IR that converts N values with target types into 1 value of the
  /// source type.
  void addTargetMaterialization(OneToNMaterializationCallbackFn &&callback) {
    oneToNTargetMaterializations.emplace_back(std::move(callback));
  }

private:
  SmallVector<OneToNMaterializationCallbackFn, 2> oneToNTargetMaterializations;
};

/// Stores a 1:N mapping of types and provides several useful accessors. This
/// class extends SignatureConversion, which already supports 1:N type mappings
/// but lacks some accessors into the mapping as well as access to the original
/// types.
class OneToNTypeMapping : public TypeConverter::SignatureConversion {
public:
  OneToNTypeMapping(TypeRange originalTypes)
      : TypeConverter::SignatureConversion(originalTypes.size()),
        originalTypes(originalTypes) {}

  using TypeConverter::SignatureConversion::getConvertedTypes;

  /// Returns the list of types that corresponds to the original type at the
  /// given index.
  TypeRange getConvertedTypes(unsigned originalTypeNo) const;

  /// Returns the list of original types.
  TypeRange getOriginalTypes() const { return originalTypes; }

  /// Returns the slice of converted values that corresponds the original value
  /// at the given index.
  ValueRange getConvertedValues(ValueRange convertedValues,
                                unsigned originalValueNo) const;

  /// Fills the given result vector with as many copies of the location of the
  /// original value as the number of values it is converted to.
  void convertLocation(Value originalValue, unsigned originalValueNo,
                       llvm::SmallVectorImpl<Location> &result) const;

  /// Fills the given result vector with as many copies of the lociation of each
  /// original value as the number of values they are respectively converted to.
  void convertLocations(ValueRange originalValues,
                        llvm::SmallVectorImpl<Location> &result) const;

  /// Returns true iff at least one type conversion maps an input type to a type
  /// that is different from itself.
  bool hasNonIdentityConversion() const;

private:
  llvm::SmallVector<Type> originalTypes;
};

/// Extends the basic RewritePattern with a type converter member and some
/// accessors to it. This is useful for patterns that are not ConversionPatterns
/// but still require access to a type converter.
class RewritePatternWithConverter : public mlir::RewritePattern {
public:
  /// Construct a conversion pattern with the given converter, and forward the
  /// remaining arguments to RewritePattern.
  template <typename... Args>
  RewritePatternWithConverter(TypeConverter &typeConverter, Args &&...args)
      : RewritePattern(std::forward<Args>(args)...),
        typeConverter(&typeConverter) {}

  /// Return the type converter held by this pattern, or nullptr if the pattern
  /// does not require type conversion.
  TypeConverter *getTypeConverter() const { return typeConverter; }

  template <typename ConverterTy>
  std::enable_if_t<std::is_base_of<TypeConverter, ConverterTy>::value,
                   ConverterTy *>
  getTypeConverter() const {
    return static_cast<ConverterTy *>(typeConverter);
  }

protected:
  /// A type converter for use by this pattern.
  TypeConverter *const typeConverter;
};

/// Specialization of PatternRewriter that OneToNConversionPatterns use. The
/// class provides additional rewrite methods that are specific to 1:N type
/// conversions.
class OneToNPatternRewriter : public PatternRewriter {
public:
  OneToNPatternRewriter(MLIRContext *context) : PatternRewriter(context) {}

  /// Replaces the results of the operation with the specified list of values
  /// mapped back to the original types as specified in the provided type
  /// mapping. That type mapping must match the replaced op (i.e., the original
  /// types must be the same as the result types of the op) and the new values
  /// (i.e., the converted types must be the same as the types of the new
  /// values).
  void replaceOp(Operation *op, ValueRange newValues,
                 const OneToNTypeMapping &resultMapping);
  using PatternRewriter::replaceOp;

  /// Applies the given argument conversion to the given block. This consists of
  /// replacing each original argument with N arguments as specified in the
  /// argument conversion and inserting unrealized casts from the converted
  /// values to the original types, which are then used in lieu of the original
  /// ones. (Eventually, applyOneToNConversion replaces these casts with a
  /// user-provided argument materialization if necessary.) This is similar to
  /// ArgConverter::applySignatureConversion but (1) handles 1:N type conversion
  /// properly and probably (2) doesn't handle many other edge cases.
  Block *applySignatureConversion(Block *block,
                                  OneToNTypeMapping &argumentConversion);
};

/// Base class for patterns with 1:N type conversions. Derived classes have to
/// overwrite the `matchAndRewrite`overlaod that provides additional information
/// for 1:N type conversions.
class OneToNConversionPattern : public RewritePatternWithConverter {
public:
  using RewritePatternWithConverter::RewritePatternWithConverter;

  /// This function has to be implemented by base classes and is called from the
  /// usual overloads. Like in normal DialectConversion, the function is
  /// provided with the converted operands (which thus have target types). Since
  /// 1:N conversion are supported, there is usually no 1:1 relationship between
  /// the original and the converted operands. Instead, the provided
  /// `operandMapping` can be used to access the converted operands that
  /// correspond to a particular original operand. Similarly, `resultMapping`
  /// is provided to help with assembling the result values (which may have 1:N
  /// correspondences as well). The function is expted to return the converted
  /// result values if the conversion succeeds and failure otherwise (in which
  /// case any modifications of the IR have to be rolled back first). The
  /// correspondance of original and converted result values needs to correspond
  /// to `resultMapping`. For both the converted operands and results, the
  /// calling overload inserts appropriate unrealized casts that produce and
  /// consume them, and replaces the uses of the results with the results of the
  /// casts. If the returned result values are the same as those of the original
  /// op, an in-place update is assumed and the result values are left as is.
  virtual LogicalResult
  matchAndRewrite(Operation *op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping &operandMapping,
                  const OneToNTypeMapping &resultMapping,
                  const SmallVector<Value> &convertedOperands) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const final;
};

/// This class is a wrapper around OneToNConversionPattern for matching against
/// instances of a particular op class.
template <typename SourceOp>
class OneToNOpConversionPattern : public OneToNConversionPattern {
public:
  OneToNOpConversionPattern(TypeConverter &typeConverter, MLIRContext *context,
                            PatternBenefit benefit = 1,
                            ArrayRef<StringRef> generatedNames = {})
      : OneToNConversionPattern(typeConverter, SourceOp::getOperationName(),
                                benefit, context, generatedNames) {}

  using OneToNConversionPattern::matchAndRewrite;

  /// Overload that derived classes have to override for their op type.
  virtual LogicalResult
  matchAndRewrite(SourceOp op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping &operandMapping,
                  const OneToNTypeMapping &resultMapping,
                  const SmallVector<Value> &convertedOperands) const = 0;

  LogicalResult
  matchAndRewrite(Operation *op, OneToNPatternRewriter &rewriter,
                  const OneToNTypeMapping &operandMapping,
                  const OneToNTypeMapping &resultMapping,
                  const SmallVector<Value> &convertedOperands) const final {
    return matchAndRewrite(cast<SourceOp>(op), rewriter, operandMapping,
                           resultMapping, convertedOperands);
  }
};

/// Main function that 1:N conversion passes should call. The patterns are
/// expected to insert unrealized casts to maintain the types of operands and
/// results, which is done automatically if the derive from
/// OneToNConversionPattern. The function replaces those that do not fold away
/// until the end of pattern application with user-provided materializations
/// from the type converter, so those have to be provided if conversions from
/// source to target types are expected to remain.
LogicalResult applyOneToNConversion(Operation *op,
                                    OneToNTypeConverter &typeConverter,
                                    const FrozenRewritePatternSet &patterns);

} // namespace iterators
} // namespace mlir

#endif // TEST_LIB_CONVERSION_ONETONTYPECONVERSION_ONETONTYPECONVERSION_H
