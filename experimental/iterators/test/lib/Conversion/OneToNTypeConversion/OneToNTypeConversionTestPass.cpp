//===- TestOneToNTypeConversion.cpp - Test 1:N type conversion utils ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OneToNTypeConversion.h"
#include "OneToNTypeConversionFunc.h"
#include "OneToNTypeConversionSCF.h"
#include "TestDialect.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::iterators;

namespace {
struct TestOneToNTypeConversion
    : public PassWrapper<TestOneToNTypeConversion, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestOneToNTypeConversion)

  TestOneToNTypeConversion() = default;
  TestOneToNTypeConversion(const TestOneToNTypeConversion &pass)
      : PassWrapper(pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<test::TestDialect>();
  }

  StringRef getArgument() const final {
    return "test-one-to-n-type-conversion";
  }

  StringRef getDescription() const final {
    return "Test pass for 1:N type conversion";
  }

  Option<bool> convertFuncOps{*this, "convert-func-ops",
                              llvm::cl::desc("Enable conversion on func ops"),
                              llvm::cl::init(false)};

  Option<bool> convertSCFOps{*this, "convert-scf-ops",
                             llvm::cl::desc("Enable conversion on scf ops"),
                             llvm::cl::init(false)};

  Option<bool> convertTupleOps{*this, "convert-tuple-ops",
                               llvm::cl::desc("Enable conversion on tuple ops"),
                               llvm::cl::init(false)};

  void runOnOperation() override;
};

} // namespace

namespace mlir {
namespace iterators {
namespace test {
void registerTestOneToNTypeConversionPass() {
  PassRegistration<TestOneToNTypeConversion>();
}
} // namespace test
} // namespace iterators
} // namespace mlir

class ConvertMakeTupleOp
    : public OneToNOpConversionPattern<::test::MakeTupleOp> {
public:
  using OneToNOpConversionPattern<
      ::test::MakeTupleOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(::test::MakeTupleOp op, PatternRewriter &rewriter,
                  const OneToNSignatureConversion &operandConversion,
                  const OneToNSignatureConversion & /*resultConversion*/,
                  const SmallVector<Value> &convertedOperands) const override {
    // Simply forward converted operands.
    return convertedOperands;
  }
};

class ConvertGetTupleElementOp
    : public OneToNOpConversionPattern<::test::GetTupleElementOp> {
public:
  using OneToNOpConversionPattern<
      ::test::GetTupleElementOp>::OneToNOpConversionPattern;

  FailureOr<SmallVector<Value>>
  matchAndRewrite(::test::GetTupleElementOp op, PatternRewriter &rewriter,
                  const OneToNSignatureConversion &operandConversion,
                  const OneToNSignatureConversion & /*resultConversion*/,
                  const SmallVector<Value> &convertedOperands) const override {
    // Construct conversion mapping for field types.
    auto stateType = op->getOperand(0).getType().cast<TupleType>();
    TypeRange originalElementTypes = stateType.getTypes();
    OneToNSignatureConversion elementConversion(originalElementTypes);
    if (failed(typeConverter->convertSignatureArgs(originalElementTypes,
                                                   elementConversion)))
      return failure();

    // Compute converted operands corresponding to original input tuple.
    ArrayRef<Value> convertedTuple =
        operandConversion.getConvertedValues(convertedOperands, 0);

    // Got those converted operands that correspond to the index-th element of
    // the original input tuple.
    size_t index = op.getIndex();
    ValueRange extractedElement =
        elementConversion.getConvertedValues(convertedTuple, index);

    return SmallVector<Value>(extractedElement);
  }
};

static void populateDecomposeTuplesPatterns(TypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      ConvertMakeTupleOp,
      ConvertGetTupleElementOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

/// Creates a sequence of `test.get_tuple_element` ops for all elements of a
/// given tuple value. If some tuple elements are, in turn, tuples, the elements
/// of those are extracted recursively such that the returned values have the
/// same types as `resultTypes.getFlattenedTypes()`.
static std::optional<SmallVector<Value>>
buildGetTupleElementOps(OpBuilder &builder, TypeRange resultTypes, Value input,
                        Location loc) {
  TupleType inputType = input.getType().dyn_cast<TupleType>();
  if (!inputType)
    return {};

  SmallVector<Value> values;
  for (auto [idx, elementType] : llvm::enumerate(inputType.getTypes())) {
    Value element = builder.create<::test::GetTupleElementOp>(
        loc, elementType, input, builder.getI32IntegerAttr(idx));
    if (auto nestedTupleType = elementType.dyn_cast<TupleType>()) {
      // Recurse if the current element is also a tuple.
      SmallVector<Type> flatRecursiveTypes;
      nestedTupleType.getFlattenedTypes(flatRecursiveTypes);
      std::optional<SmallVector<Value>> resursiveValues =
          buildGetTupleElementOps(builder, flatRecursiveTypes, element, loc);
      if (!resursiveValues.has_value())
        return {};
      values.append(resursiveValues.value());
    } else {
      values.push_back(element);
    }
  }
  return values;
}

/// Creates a `test.make_tuple` op out of the given inputs building a tuple of
/// type `resultType`. If that type is nested, each nested tuple is built
/// recursively with another `test.make_tuple` op.
static std::optional<Value> buildMakeTupleOp(OpBuilder &builder,
                                             TupleType resultType,
                                             ValueRange inputs, Location loc) {
  // Build one value for each element at this nesting level.
  SmallVector<Value> elements;
  elements.reserve(resultType.getTypes().size());
  ValueRange::iterator inputIt = inputs.begin();
  for (Type elementType : resultType.getTypes()) {
    if (auto nestedTupleType = elementType.dyn_cast<TupleType>()) {
      // Determine how many input values are needed for the nested elements of
      // the nested TupleType and advance inputIt by that number.
      // TODO: We only need the *number* of nested types, not the types itself.
      //       Maybe it's worth adding a more efficient overload?
      SmallVector<Type> nestedFlattenedTypes;
      nestedTupleType.getFlattenedTypes(nestedFlattenedTypes);
      size_t numNestedFlattenedTypes = nestedFlattenedTypes.size();
      ValueRange nestedFlattenedelements(inputIt,
                                         inputIt + numNestedFlattenedTypes);
      inputIt += numNestedFlattenedTypes;

      // Recurse on the values for the nested TupleType.
      std::optional<Value> res = buildMakeTupleOp(builder, nestedTupleType,
                                                  nestedFlattenedelements, loc);
      if (!res.has_value())
        return {};

      // The tuple constructed by the conversion is the element value.
      elements.push_back(res.value());
    } else {
      // Base case: take one input as is.
      elements.push_back(*inputIt++);
    }
  }

  // Assemble the tuple from the elements.
  return builder.create<::test::MakeTupleOp>(loc, resultType, elements);
}

void TestOneToNTypeConversion::runOnOperation() {
  ModuleOp module = getOperation();
  auto *context = &getContext();

  // Assemble type converter.
  OneToNTypeConverter typeConverter;

  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [](TupleType tupleType, SmallVectorImpl<Type> &types) {
        tupleType.getFlattenedTypes(types);
        return success();
      });

  typeConverter.addArgumentMaterialization(buildMakeTupleOp);
  typeConverter.addSourceMaterialization(buildMakeTupleOp);
  typeConverter.addTargetMaterialization(buildGetTupleElementOps);

  // Assemble patterns.
  RewritePatternSet patterns(context);
  if (convertTupleOps)
    populateDecomposeTuplesPatterns(typeConverter, patterns);
  if (convertFuncOps)
    populateFuncTypeConversionPatterns(typeConverter, patterns);
  if (convertSCFOps)
    populateSCFTypeConversionPatterns(typeConverter, patterns);

  // Run conversion.
  if (failed(applyOneToNConversion(module, typeConverter, std::move(patterns))))
    return signalPassFailure();
}
