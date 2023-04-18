//===- DecomposeTupless.cpp - Pass Implementation ----------------*- C++
//-*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Tuple/Transforms/DecomposeTuples.h"

#include "mlir/Dialect/Func/Transforms/OneToNFuncConversions.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/OneToNTypeConversion.h"
#include "structured/Dialect/Tuple/IR/Tuple.h"
#include "structured/Dialect/Tuple/Transforms/Passes.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
#define GEN_PASS_CLASSES
#include "structured/Dialect/Tuple/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::structured;

using InputMapping = OneToNTypeMapping::InputMapping;

namespace {

class DecomposeTuplesTypeConverter : public OneToNTypeConverter {
public:
  DecomposeTuplesTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](TupleType type, SmallVectorImpl<Type> &results) {
      type.getFlattenedTypes(results);
      return success();
    });
  }
};

class DecomposeFromElementsOp
    : public OneToNOpConversionPattern<FromElementsOp> {
public:
  using OneToNOpConversionPattern<FromElementsOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(FromElementsOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Simply forward converted operands.
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());

    return success();
  }
};

class DecomposeToElementsOp : public OneToNOpConversionPattern<ToElementsOp> {
public:
  using OneToNOpConversionPattern<ToElementsOp>::OneToNOpConversionPattern;

  LogicalResult
  matchAndRewrite(ToElementsOp op, OpAdaptor adaptor,
                  OneToNPatternRewriter &rewriter) const override {
    // Simply forward converted operands.
    rewriter.replaceOp(op, adaptor.getFlatOperands(),
                       adaptor.getResultMapping());

    return success();
  }
};

} // namespace

void structured::populateDecomposeTuplesPatterns(TypeConverter &typeConverter,
                                                 RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      DecomposeFromElementsOp,
      DecomposeToElementsOp
      // clang-format on
      >(typeConverter, patterns.getContext());
}

/// Creates IR that builds `FromElementsOp`s to assemble a value of the given,
/// portentially recursive tuple type from the given range of inputs. This is
/// meant to be used as argument and source materializations for tuple
/// decomposition.
///
/// The implementation of this function is recursive, which should be avoided in
/// the LLVM code base. However, the recursion is bounded by the nesting depth
/// of the iterator state types, which should make this an acceptable exception.
static std::optional<Value> buildFromElementsOp(OpBuilder &builder,
                                                TypeConverter &typeConverter,
                                                Type type, ValueRange inputs,
                                                Location loc) {
  auto tupleType = type.dyn_cast<TupleType>();
  if (!tupleType)
    return {};

  // Create mapping for elements of inputs.
  TypeRange originalInputTypes = tupleType.getTypes();
  OneToNTypeMapping inputMapping(originalInputTypes);
  if (failed(
          typeConverter.convertSignatureArgs(originalInputTypes, inputMapping)))
    return {};

  // Assemble element values at this nesting level.
  SmallVector<Value> operands;
  for (auto [i, elementType] : llvm::enumerate(originalInputTypes)) {
    // Element is a nested tuple: recursively build back the tuple.
    ValueRange elementValues = inputMapping.getConvertedValues(inputs, i);
    if (elementType.isa<TupleType>()) {
      std::optional<Value> createdTuple = buildFromElementsOp(
          builder, typeConverter, elementType, elementValues, loc);
      if (!createdTuple.has_value() || !createdTuple.value())
        return {};
      operands.push_back(createdTuple.value());
      continue;
    }

    // Any other type: take as is.
    assert(elementValues.size() == 1);
    Value operand = elementValues.front();
    operands.push_back(operand);
  }

  // Build the tuple from its elements.
  auto createStateOp = builder.create<FromElementsOp>(loc, type, operands);
  assert(createStateOp->getNumResults() == 1);
  return createStateOp->getResult(0);
}

/// Creates IR that extracts the elements of the given input tuple recursively
/// using `ToElementOp`s. This is meant ti be used as target conversion for
/// tuple decomposition.
///
/// The implementation of this function is recursive, which should be avoided in
/// the LLVM code base. However, the recursion is bounded by the nesting depth
/// of the iterator state types, which should make this an acceptable exception.
static std::optional<SmallVector<Value>>
buildToElementsOp(OpBuilder &builder, TypeConverter &typeConverter,
                  TypeRange resultTypes, Value input, Location loc) {
  auto tupleType = input.getType().dyn_cast<TupleType>();
  if (!tupleType)
    return {};

  // Create mapping for elements of inputs.
  TypeRange originalElementTypes = tupleType.getTypes();
  OneToNTypeMapping elementMapping(originalElementTypes);
  if (failed(typeConverter.convertSignatureArgs(originalElementTypes,
                                                elementMapping)))
    return {};

  // Extract elements at this level.
  auto toElements =
      builder.create<ToElementsOp>(loc, originalElementTypes, input);

  // Assemble final values, recursing where necessary.
  SmallVector<Value> results;
  for (auto [i, nestedElement] : llvm::enumerate(toElements.getResults())) {
    // Nested element is a tuple: use results of recursive call.
    if (nestedElement.getType().isa<TupleType>()) {
      TypeRange nestedResultTypes = elementMapping.getConvertedTypes(i);
      std::optional<SmallVector<Value>> nestedValues = buildToElementsOp(
          builder, typeConverter, nestedResultTypes, nestedElement, loc);
      if (!nestedValues.has_value())
        return {};
      results.append(nestedValues->begin(), nestedValues->end());
    } else {
      // Any other element: use as is.
      results.push_back(nestedElement);
    }
  }

  return results;
}

namespace {

struct DecomposeTuplesPass : public DecomposeTuplesBase<DecomposeTuplesPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    // Assemble type convert with materializations.
    DecomposeTuplesTypeConverter typeConverter;
    auto buildFromElementsOpHelper = [&](OpBuilder &builder, Type type,
                                         ValueRange inputs, Location loc) {
      return buildFromElementsOp(builder, typeConverter, type, inputs, loc);
    };
    typeConverter.addArgumentMaterialization(buildFromElementsOpHelper);
    typeConverter.addSourceMaterialization(buildFromElementsOpHelper);
    auto buildToElementsOpHelper = [&](OpBuilder &builder,
                                       TypeRange resultTypes, Value input,
                                       Location loc) {
      return buildToElementsOp(builder, typeConverter, resultTypes, input, loc);
    };
    typeConverter.addTargetMaterialization(buildToElementsOpHelper);

    // Assemble patterns.
    RewritePatternSet patterns(context);
    populateDecomposeTuplesPatterns(typeConverter, patterns);
    if (convertFuncOps)
      populateFuncTypeConversionPatterns(typeConverter, patterns);
    if (convertSCFOps)
      scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);

    // Run conversion.
    if (failed(applyPartialOneToNConversion(module, typeConverter,
                                            std::move(patterns))))
      return signalPassFailure();
  };
};

} // namespace

std::unique_ptr<Pass> mlir::createDecomposeTuplesPass() {
  return std::make_unique<DecomposeTuplesPass>();
}
