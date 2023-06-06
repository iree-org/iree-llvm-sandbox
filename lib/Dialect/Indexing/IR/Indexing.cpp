//===-- Indexing.cpp - Indexing dialect -------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Indexing/IR/Indexing.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

#include <numeric>

using namespace mlir;
using namespace mlir::indexing;

//===----------------------------------------------------------------------===//
// Indexing dialect
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Indexing/IR/IndexingOpsDialect.cpp.inc"

void IndexingDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "structured/Dialect/Indexing/IR/IndexingOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "structured/Dialect/Indexing/IR/IndexingOpsTypes.cpp.inc"
      >();
}

Operation *IndexingDialect::materializeConstant(OpBuilder &builder,
                                                Attribute value, Type type,
                                                Location loc) {
  auto op = arith::ConstantOp::materialize(builder, value, type, loc);
  if (!op)
    emitError(loc, "Couldn't materialize constant array.");
  return op;
}

//===----------------------------------------------------------------------===//
// Indexing operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES

#include "structured/Dialect/Indexing/IR/IndexingOps.cpp.inc"

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

LogicalResult GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  ArrayRef<int64_t> gather_dims =
      attributes.get("gather_dims").cast<mlir::DenseI64ArrayAttr>();
  RankedTensorType expectedResultType = mlir::tensor::GatherOp::inferResultType(
      // source
      operands[0].getType().cast<RankedTensorType>(),
      // indices
      operands[1].getType().cast<RankedTensorType>(), gather_dims,
      /*rankReduced=*/true);
  inferredReturnTypes.assign({expectedResultType});
  return success();
}

bool GatherOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

bool ScatterOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

//===----------------------------------------------------------------------===//
// ConcatenateOp
//===----------------------------------------------------------------------===//

LogicalResult ConcatenateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  auto dimension = attributes.get("dimension").cast<IntegerAttr>().getInt();
  auto sourceType = operands[0].getType().cast<RankedTensorType>();
  SmallVector<int64_t> resultShape(sourceType.getShape());
  std::for_each(
      operands.begin() + 1, operands.end(),
      [&resultShape, dimension](const Value &v) {
        resultShape[dimension] +=
            v.getType().cast<RankedTensorType>().getShape()[dimension];
      });
  inferredReturnTypes.assign(
      {RankedTensorType::Builder(sourceType).setShape(resultShape)});
  return success();
}

bool ConcatenateOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

//===----------------------------------------------------------------------===//
// ARangeOp
//===----------------------------------------------------------------------===//

// numpy semantics
int64_t getARangeLen(int64_t start, int64_t stop, int64_t step) {
  auto len = floorDiv((stop - start), step) + 1;
  if ((stop - start) % step == 0)
    len--;
  return len;
}

LogicalResult ARangeOp::inferReturnTypes(
    MLIRContext *context, std::optional<mlir::Location> location,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<mlir::Type> &inferredReturnTypes) {
  if (!operands.empty()) {
    inferredReturnTypes.assign({RankedTensorType::get(
        {ShapedType::kDynamic}, IndexType::get(context))});
  } else if (!attributes.empty() &&
             attributes.contains(ARangeOp::getStartAttrAttrName(context)) &&
             attributes.contains(ARangeOp::getStopAttrAttrName(context)) &&
             attributes.contains(ARangeOp::getStepAttrAttrName(context))) {
    auto start = attributes.get(ARangeOp::getStartAttrAttrName(context))
                     .cast<IntegerAttr>()
                     .getInt();
    auto stop = attributes.get(ARangeOp::getStopAttrAttrName(context))
                    .cast<IntegerAttr>()
                    .getInt();
    auto step = attributes.get(ARangeOp::getStepAttrAttrName(context))
                    .cast<IntegerAttr>()
                    .getInt();
    inferredReturnTypes.assign({RankedTensorType::get(
        {getARangeLen(start, stop, step)}, IndexType::get(context))});
  } else {
    return failure();
  }
  return success();
}

bool ARangeOp::isCompatibleReturnTypes(TypeRange l, TypeRange r) {
  if (l.size() != r.size() || l.size() != 1)
    return false;
  return succeeded(verifyCompatibleShape(l[0], r[0]));
}

LogicalResult ARangeOp::verify() {
  if (getStartAttr() && getStart())
    return emitError(
        "Start can only be provided as either an attribute or an operand");
  if (getStopAttr() && getStop())
    return emitError(
        "Stop can only be provided as either an attribute or an operand");
  if (getStepAttr() && getStep())
    return emitError(
        "Step can only be provided as either an attribute or an operand");
  if (getStartAttr() && getStartAttr().value().getSExtValue() < 0)
    return emitError("Start must be >= 0.");
  if (getStopAttr() && getStopAttr().value().getSExtValue() < 1)
    return emitError("Stop must be > 0.");
  if (getStepAttr() && getStepAttr().value().getSExtValue() < 1)
    return emitError("Step must be > 0.");
  if (getStartAttr() and getStopAttr() and
      getStopAttr().value().getSExtValue() -
              getStartAttr().value().getSExtValue() <=
          1)
    return emitError("Stop - Start must be > 1");
  return success();
}

namespace {

struct ARangeOpPattern : public RewritePattern {
  ARangeOpPattern(MLIRContext *context)
      : RewritePattern(ARangeOp::getOperationName(), 1, context) {}

  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  LogicalResult match(Operation *op) const override {
    if (!op->getOperands().empty() &&
        llvm::any_of(op->getOperands(), [](Value op) {
          arith::ConstantOp arithCst =
              dyn_cast<arith::ConstantOp>(op.getDefiningOp());
          return arithCst && arithCst.getValue().getType().isa<IndexType>();
        })) {
      return success();
    }
    return failure();
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
    auto arangeOp = cast<ARangeOp>(op);
    SmallVector<NamedAttribute, 4> attributes;
    SmallVector<Value, 3> operands;
    SmallVector<int32_t, 3> segmentSizes{1, 1, 1};
    SmallVector<StringAttr> attrs = {arangeOp.getStartAttrAttrName(),
                                     arangeOp.getStopAttrAttrName(),
                                     arangeOp.getStepAttrAttrName()};
    SmallVector<Value> opers = {arangeOp.getStart(), arangeOp.getStop(),
                                arangeOp.getStep()};
    for (const auto &[index, tuple] :
         llvm::enumerate(llvm::zip(opers, attrs))) {
      auto val = std::get<0>(tuple);
      auto attrName = std::get<1>(tuple);
      if (!val) {
        attributes.push_back({attrName, arangeOp->getAttr(attrName)});
        segmentSizes[index] = 0;
        continue;
      }

      auto namedAttr = arangeOp->getAttrOfType<IntegerAttr>(attrName);
      arith::ConstantOp arithCst =
          dyn_cast<arith::ConstantOp>(val.getDefiningOp());

      if (!arithCst) {
        operands.push_back(val);
        continue;
      }

      if (auto cstAttr = arithCst.getValue().dyn_cast<IntegerAttr>()) {
        if (namedAttr && namedAttr.getInt() != cstAttr.getInt())
          arangeOp.emitError(llvm::Twine("Ambiguous value for ") +
                             attrName.getValue());
        attributes.push_back({attrName, cstAttr});
        segmentSizes[index] = 0;
        continue;
      }
      operands.push_back(val);
    }

    assert(operands.size() + attributes.size() == 3 &&
           "wrong number of operands and attributes");
    assert(operands.size() ==
               std::reduce(segmentSizes.begin(), segmentSizes.end()) &&
           "expected number of non-zero segments to equal number of operands.");

    auto attr = rewriter.getDenseI32ArrayAttr(segmentSizes);
    attributes.push_back({arangeOp.getOperandSegmentSizesAttrName(), attr});
    if (arangeOp.getNofold())
      attributes.push_back(
          {arangeOp.getNofoldAttrName(), arangeOp.getNofoldAttr()});
    rewriter.replaceOpWithNewOp<ARangeOp>(arangeOp, arangeOp->getResultTypes(),
                                          operands, attributes);
  }
};

} // namespace

void ARangeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add<ARangeOpPattern>(context);
}

OpFoldResult ARangeOp::fold(FoldAdaptor adaptor) {
  if (!adaptor.getStartAttr() || !adaptor.getStopAttr() ||
      !adaptor.getStepAttr() || adaptor.getNofold())
    return {};

  int64_t start = adaptor.getStartAttr().value().getSExtValue(),
          stop = adaptor.getStopAttr().value().getSExtValue(),
          step = adaptor.getStepAttr().value().getSExtValue();
  std::vector<int64_t> arange;
  auto len = getARangeLen(start, stop, step);
  for (int64_t i = start; i < stop; i += step) {
    arange.push_back(i);
  }
  auto type = RankedTensorType::get({len}, IndexType::get(getContext()));
  return DenseElementsAttr::get(type, ArrayRef(arange));
}

//===----------------------------------------------------------------------===//
// MeshGridOp
//===----------------------------------------------------------------------===//

LogicalResult MeshGridOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  llvm::SmallVector<int64_t> shape;
  for (const Value &op : operands) {
    auto opShape = op.getType().cast<RankedTensorType>();
    if (opShape.getRank() != 1) {
      op.getDefiningOp()->emitError("MeshGrid operand must be 1d.");
      return failure();
    }
    shape.push_back(opShape.getDimSize(0));
  }
  shape.push_back(operands.size());
  inferredReturnTypes.assign(
      {RankedTensorType::get(shape, IndexType::get(context))});
  return success();
}

LogicalResult MeshGridOp::verify() {
  if (!llvm::all_of(getOperandTypes(), [](Type t) {
        if (auto r = t.dyn_cast<RankedTensorType>())
          return r.hasRank() and r.getRank() == 1;
        return false;
      }))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Indexing types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES

#include "structured/Dialect/Indexing/IR/IndexingOpsTypes.cpp.inc"
