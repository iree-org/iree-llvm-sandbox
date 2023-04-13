//===- DecomposeIteratorStates.cpp - Pass Implementation --------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Iterators/Transforms/DecomposeIteratorStates.h"

#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Dialect/Iterators/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/OneToNTypeConversion.h"

namespace mlir {
#define GEN_PASS_CLASSES
#include "iterators/Dialect/Iterators/Transforms/Passes.h.inc"
} // namespace mlir

// using namespace iterators;
using namespace mlir;
using namespace mlir::iterators;

namespace {

class DecomposeStatesTypeConverter : public OneToNTypeConverter {
public:
  DecomposeStatesTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([&](StateType type, SmallVectorImpl<Type> &results) {
      return decomposeStateType(type, results);
    });
  }

private:
  /// Maps a StateType to the types of its fields.
  Optional<LogicalResult> decomposeStateType(StateType type,
                                             SmallVectorImpl<Type> &results) {
    for (Type fieldType : type.getFieldTypes()) {
      if (failed(convertTypes(fieldType, results)))
        return failure();
    }
    return success();
  }
};

class DecomposeCreateStateOp : public OneToNOpConversionPattern<CreateStateOp> {
public:
  using OneToNOpConversionPattern<CreateStateOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(CreateStateOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Simply replace the current op with the converted operands.
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());
    return success();
  }
};

class DecomposeInsertValueOp : public OneToNOpConversionPattern<InsertValueOp> {
public:
  using OneToNOpConversionPattern<InsertValueOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(InsertValueOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Construct conversion mapping for field types.
    auto stateType = op.getState().getType().cast<StateType>();
    TypeRange originalFieldTypes = stateType.getFieldTypes();
    OneToNTypeMapping fieldMapping(originalFieldTypes);
    if (failed(typeConverter->convertSignatureArgs(originalFieldTypes,
                                                   fieldMapping)))
      return failure();

    // Extract converted operands.
    ValueRange convertedState = adaptor.getState();
    ValueRange convertedValue = adaptor.getValue();

    // Compose new state fields from unchanged and inserted ones.
    size_t index = op.getIndex().getZExtValue();
    SmallVector<Value> updatedState;
    for (size_t i = 0; i < stateType.getFieldTypes().size(); i++) {
      ValueRange field = fieldMapping.getConvertedValues(convertedState, i);
      if (index == i) {
        assert(field.getTypes() == ValueRange{convertedValue}.getTypes());
        updatedState.append(convertedValue.begin(), convertedValue.end());
      } else {
        updatedState.append(field.begin(), field.end());
      }
    }

    // Replace original op with new state fields.
    rewriter.replaceOp(op, updatedState, adaptor.getResultMapping());
    return success();
  }
};

class DecomposeExtractValueOp
    : public OneToNOpConversionPattern<ExtractValueOp> {
public:
  using OneToNOpConversionPattern<ExtractValueOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(ExtractValueOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Construct conversion mapping for field types.
    auto stateType = op.getState().getType().cast<StateType>();
    TypeRange originalFieldTypes = stateType.getFieldTypes();
    OneToNTypeMapping fieldMapping(originalFieldTypes);
    if (failed(typeConverter->convertSignatureArgs(originalFieldTypes,
                                                   fieldMapping)))
      return failure();

    // Extract converted operands.
    ValueRange convertedState = adaptor.getState();

    // Return extracted value.
    size_t index = op.getIndex().getZExtValue();
    ValueRange extractedValue =
        fieldMapping.getConvertedValues(convertedState, index);

    // Replace original op with extracted field value.
    rewriter.replaceOp(op, extractedValue, adaptor.getResultMapping());
    return success();
  }
};

} // namespace

void iterators::populateDecomposeIteratorStatesPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      DecomposeExtractValueOp,
      DecomposeInsertValueOp,
      DecomposeCreateStateOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

/// Creates IR that builds `CreateStateOp`s to assemble an iterator state of the
/// given, portentially recursive state type from the given range of value. This
/// is meant to be used as argument and source materializations for iterator
/// state decomposition.
///
/// The implementation of this function is recursive, which should be avoided in
/// the LLVM code base. On the one side, the recursion is bounded by the nesting
/// depth of the iterator state types, which might make this an acceptable
/// exception. On the other side, the nesting depth of the states currently
/// depends on the length of SSA use-def chains, which may be large. That,
/// however, is a problem with the overall design of the current lowering and
/// not specific to state decomposition. If/when that is fixed, the problem here
/// will either go away comletely or be fixed as a consequence.
static std::optional<Value> buildCreateStateOp(OpBuilder &builder,
                                               TypeConverter &typeConverter,
                                               Type type, ValueRange inputs,
                                               Location loc) {
  auto stateType = type.dyn_cast<StateType>();
  if (!stateType)
    return {};

  SmallVector<Value> operands;
  operands.reserve(stateType.getFieldTypes().size());
  ValueRange::iterator inputsIt = inputs.begin();
  for (Type fieldType : stateType.getFieldTypes()) {
    Value operand;
    if (fieldType.isa<StateType>()) {
      SmallVector<Type> nestedFieldTypes;
      if (failed(typeConverter.convertType(fieldType, nestedFieldTypes)))
        return {};
      ValueRange nestedFields{inputsIt, inputsIt + nestedFieldTypes.size()};
      std::optional<Value> createdState = buildCreateStateOp(
          builder, typeConverter, fieldType, nestedFields, loc);
      if (!createdState.has_value() || !createdState.value())
        return {};
      operand = createdState.value();
    } else {
      operand = *inputsIt;
    }
    operands.push_back(operand);
  }
  auto createStateOp = builder.create<CreateStateOp>(loc, type, operands);
  assert(createStateOp->getNumResults() == 1);
  return createStateOp->getResult(0);
}

/// Creates IR that builds `ExtractValueOp`s to extract the (potentially nested)
/// constituent values from the given iterator state. This is meant to be used
/// as target materializations for iterator state decomposition.
///
/// The implementation of this function is recursive, which should be avoided in
/// the LLVM code base. On the one side, the recursion is bounded by the nesting
/// depth of the iterator state types, which might make this an acceptable
/// exception. On the other side, the nesting depth of the states currently
/// depends on the length of SSA use-def chains, which may be large. That,
/// however, is a problem with the overall design of the current lowering and
/// not specific to state decomposition. If/when that is fixed, the problem here
/// will either go away comletely or be fixed as a consequence.
static std::optional<SmallVector<Value>>
buildExtractValueOps(OpBuilder &builder, TypeConverter &typeConverter,
                     TypeRange resultTypes, Value input, Location loc) {
  auto stateType = input.getType().dyn_cast<StateType>();
  if (!stateType)
    return {};

  SmallVector<Value> extractedValues;
  for (auto [idx, fieldType] : llvm::enumerate(stateType.getFieldTypes())) {
    Value extractedValue =
        builder.create<ExtractValueOp>(loc, input, builder.getIndexAttr(idx));

    // If the value isn't a nested state, we can take it as is.
    if (!fieldType.isa<StateType>()) {
      extractedValues.push_back(extractedValue);
      continue;
    }

    // For states, we recurse to extract the nested fields.
    SmallVector<Type> nestedResultTypes;
    if (failed(typeConverter.convertType(extractedValue.getType(),
                                         nestedResultTypes)))
      return {};
    std::optional<SmallVector<Value>> maybeResults = buildExtractValueOps(
        builder, typeConverter, nestedResultTypes, extractedValue, loc);
    if (!maybeResults)
      return {};

    extractedValues.append(*maybeResults);
  }

  return extractedValues;
}

namespace {

struct DecomposeIteratorStatesPass
    : public DecomposeIteratorStatesBase<DecomposeIteratorStatesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    DecomposeStatesTypeConverter typeConverter;
    auto buildCreateStateOpHelper = [&](OpBuilder &builder, Type type,
                                        ValueRange inputs, Location loc) {
      return buildCreateStateOp(builder, typeConverter, type, inputs, loc);
    };
    typeConverter.addArgumentMaterialization(buildCreateStateOpHelper);
    typeConverter.addSourceMaterialization(buildCreateStateOpHelper);
    auto buildExtractValueOpsHelper = [&](OpBuilder &builder,
                                          TypeRange resultTypes, Value input,
                                          Location loc) {
      return buildExtractValueOps(builder, typeConverter, resultTypes, input,
                                  loc);
    };
    typeConverter.addTargetMaterialization(buildExtractValueOpsHelper);

    RewritePatternSet patterns(context);
    populateDecomposeIteratorStatesPatterns(typeConverter, patterns);
    populateFuncTypeConversionPatterns(typeConverter, patterns);
    scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
    if (failed(applyPartialOneToNConversion(module, typeConverter,
                                            std::move(patterns))))
      return signalPassFailure();
  };
};

} // namespace

std::unique_ptr<Pass> mlir::createDecomposeIteratorStatesPass() {
  return std::make_unique<DecomposeIteratorStatesPass>();
}
