//===- TilingExternalModels.cpp - External models for TilingInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgExt/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-ext-tiling"

using namespace mlir;
using namespace mlir::linalg;

static Value getAsValue(OpBuilder &b, Location loc, OpFoldResult ofr) {
  if (auto v = ofr.dyn_cast<Value>())
    return v;
  return b.create<arith::ConstantIndexOp>(
      loc, ofr.get<Attribute>().cast<IntegerAttr>().getInt());
}
static SmallVector<Value> getAsValues(OpBuilder &b, Location loc,
                                      ArrayRef<OpFoldResult> ofrs) {
  SmallVector<Value> vals;
  vals.reserve(ofrs.size());
  for (auto ofr : ofrs)
    vals.push_back(getAsValue(b, loc, ofr));
  return vals;
}

static SmallVector<Value, 4> makeTiledInputShapes(OpBuilder &b, Location loc,
                                                  LinalgOp linalgOp,
                                                  ArrayRef<Value> valuesToTile,
                                                  Value iv, Value tileSize,
                                                  ArrayRef<Value> sizeBounds) {
  assert(static_cast<int64_t>(valuesToTile.size()) == linalgOp.getNumInputs() &&
         "expected one value to tile for every operand");

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> tileSizes(sizeBounds.size(), zero);
  tileSizes[0] = tileSize;

  // Construct (potentially temporary) mins and maxes on which to apply maps
  // that define tile subshapes.
  SmallVector<Value> lbs = computeTileOffsets(b, loc, iv, tileSizes);
  SmallVector<Value> subShapeSizes =
      computeTileSizes(b, loc, iv, tileSizes, sizeBounds);

  SmallVector<Value, 4> tiledShapes;
  tiledShapes.reserve(valuesToTile.size());
  for (OpOperand *opOperand : linalgOp.getInputOperands()) {
    Value shapedOp = valuesToTile[opOperand->getOperandNumber()];
    LLVM_DEBUG(llvm::dbgs() << "makeTiledShapes: for operand " << shapedOp);
    AffineMap map = linalgOp.getTiedIndexingMap(opOperand);
    LLVM_DEBUG(llvm::dbgs() << ": tiled: figure out subshape...\n");
    tiledShapes.push_back(makeTiledShape(b, loc, shapedOp, tileSizes, map, lbs,
                                         sizeBounds, subShapeSizes));
  }

  return tiledShapes;
}

namespace {

/// External model implementation of TilingInterface for LinalgOps. This is
/// templated on the actual Linalg named op for now since the registration of
/// the external model requires the original operation.
template <typename LinalgOpTy>
struct LinalgOpTilingInterface
    : public TilingInterface::ExternalModel<LinalgOpTilingInterface<LinalgOpTy>,
                                            LinalgOpTy> {
  SmallVector<Value> getDestinationOperands(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    return linalgOp.getOutputOperands();
  }

  SmallVector<StringRef> getLoopIteratorTypes(Operation *op) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    SmallVector<StringRef> iteratorTypes;
    iteratorTypes.reserve(linalgOp.iterator_types().size());
    for (Attribute iteratorAttr : linalgOp.iterator_types()) {
      iteratorTypes.push_back(iteratorAttr.cast<StringAttr>().getValue());
    }
    return iteratorTypes;
  }

  SmallVector<Range> getIterationDomain(Operation *op, OpBuilder &b) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    return linalgOp.createLoopRanges(b, op->getLoc());
  }

  SmallVector<Operation *>
  getTiledImplementation(Operation *op, OpBuilder &b, ValueRange tiledDest,
                         ArrayRef<OpFoldResult> offsets,
                         ArrayRef<OpFoldResult> sizes,
                         bool tileDestOperands) const {
    LinalgOp linalgOp = cast<LinalgOp>(op);
    Location loc = op->getLoc();
    AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
    auto allShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
    if (!shapeSizesToLoopsMap)
      return {};

    SmallVector<Value> tileOffsets = getAsValues(b, loc, offsets);
    SmallVector<Value> tileSizes = getAsValues(b, loc, sizes);
    SmallVector<Value> sizeBounds =
        applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
    SmallVector<Value> valuesToTile = linalgOp.getInputOperands();
    SmallVector<Value> tiledOperands;
    if (tileDestOperands) {
      // Append the outputs then tile both the inputs and outputs.
      valuesToTile.append(tiledDest.begin(), tiledDest.end());
      tiledOperands = makeTiledShapes(b, loc, linalgOp, valuesToTile,
                                      tileOffsets, tileSizes, sizeBounds);
    } else {
      // Only tile the inputs, then apped the outputs.
      tiledOperands = makeTiledInputShapes(b, loc, linalgOp, valuesToTile,
                                           tileOffsets.front(),
                                           tileSizes.front(), sizeBounds);
      tiledOperands.append(tiledDest.begin(), tiledDest.end());
    }

    return {linalgOp.clone(b, loc, tiledDest.getTypes(), tiledOperands)};
  }
};
} // namespace

template <typename OpType>
void registerOne(DialectRegistry &registry) {
  registry.addOpInterface<OpType, LinalgOpTilingInterface<OpType>>();
}

/// Variadic helper function.
template <typename... OpTypes>
void registerAll(DialectRegistry &registry) {
  // FIXME: In c++17 this can be simplified by using 'fold expressions'.
  (void)std::initializer_list<int>{0, (registerOne<OpTypes>(registry), 0)...};
}

#define GET_OP_LIST

void mlir::linalg_ext::registerTilingInterfaceExternalModels(
    DialectRegistry &registry) {
  registerOne<linalg::GenericOp>(registry);
  registerAll<
#include "mlir/Dialect/Linalg/IR/LinalgStructuredOps.cpp.inc"
      >(registry);
}
