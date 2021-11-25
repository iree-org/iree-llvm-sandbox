//===- VectorMaskingUtils.cpp - Utilities for vector masking --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorMaskingUtils.h"

#include "Dialects/VectorExt/VectorExtOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::vector;
using namespace mlir::vector_ext;

using ExprList = ArrayRef<ArrayRef<AffineExpr>>;

/// Move the operation range `opRange` before operation `dest`.
static void moveOperationsBefore(llvm::iterator_range<Block::iterator> opRange,
                                 Operation *dest) {
  if (opRange.empty())
    return;
  dest->getBlock()->getOperations().splice(
      Block::iterator(dest), opRange.begin()->getBlock()->getOperations(),
      opRange.begin(), opRange.end());
}

/// Iterate over all the definitions produced by the operation range `opRange`
/// and gather those with users outside the range.
static void
getDefsWithExternalUses(llvm::iterator_range<Block::iterator> opRange,
                        SmallVectorImpl<Value> &defsWithExternalUses,
                        SmallVectorImpl<Type> &defTypes) {
  SmallPtrSet<Operation *, 8> rangeSet;
  for (Operation &op : opRange)
    rangeSet.insert(&op);

  for (Operation &op : llvm::make_early_inc_range(opRange))
    for (Value res : op.getResults())
      for (Operation *user : res.getUsers())
        if (rangeSet.count(user) == 0) {
          defsWithExternalUses.push_back(res);
          defTypes.push_back(res.getType());
          break;
        }
}

/// Introduce a vector.predicate op that encloses the loop body of the tiled
/// loop. The predicate is computed based on the provided vectorization factor.
static void createVectorPredicateOp(TiledLoopOp loopOp, int64_t vecFactor) {
  // Compute the range of operations that will be moved within vector.predicate
  // and the definitions within the range with external users.
  Block &bodyBlock = loopOp.getLoopBody().front();
  auto opsToMove =
      llvm::make_range(bodyBlock.begin(), Block::iterator(bodyBlock.back()));

  SmallVector<Value, 8> defsWithExternalUses;
  SmallVector<Type, 8> defTypes;
  getDefsWithExternalUses(opsToMove, defsWithExternalUses, defTypes);

  // Initialize a builder to insert vector.predicate at the beginning of the
  // loop body.
  Location loc = loopOp.getLoc();
  MLIRContext *context = loopOp.getContext();
  OpBuilder builder(loopOp);
  builder.setInsertionPointToStart(&bodyBlock);

  // Generate the predicate to used by the vector.predicate operation:
  //   %min = affine.min affine_map<(d0)[s0] -> (vecFactor, -d0 + s0)>(%iv)[%ub]
  //   %vec_pred = vector.create_mask %min : vector<8xi1>
  // TODO: This is probably not the most efficient way to generate the vector
  // predicate. Consider using the vector IV.
  AffineExpr i, j;
  bindDims(context, i, j);
  SmallVector<AffineMap, 4> maps = AffineMap::inferFromExprList(
      ExprList{{builder.getAffineConstantExpr(vecFactor), i - j}});
  auto minOp = builder.create<AffineMinOp>(
      loc, loopOp.step()[0].getType(), maps[0],
      ValueRange{loopOp.upperBound()[0], loopOp.getInductionVars()[0]});

  auto maskType = VectorType::get({vecFactor}, builder.getI1Type());
  auto predicate =
      builder.create<CreateMaskOp>(loc, maskType, ValueRange{minOp});

  // Generate the vector.predicate operation and move the range [`startOp`,
  // `endOp`] of operations within its truePredicateRegion. We have to rewire
  // the def-use chain for those definitions within the range that have external
  // uses. Those uses are rewired to the results of the vector.predicate.
  auto vecPredOp = builder.create<PredicateOp>(loc, defTypes, predicate);
  assert(defsWithExternalUses.size() == vecPredOp.getNumResults() &&
         "Expected same size");
  for (auto &en : llvm::enumerate(defsWithExternalUses))
    en.value().replaceAllUsesWith(vecPredOp.getResult(en.index()));

  Operation *truePredTerminator =
      &vecPredOp.truePredicateRegion().front().back();
  moveOperationsBefore(opsToMove, truePredTerminator);

  // The existing terminator of TruePredicateRegion doesn't yield any value.
  // Replace it with a new terminator that returns the definitions with external
  // uses that we just moved.
  if (vecPredOp.getNumResults() > 0) {
    builder.setInsertionPoint(truePredTerminator);
    builder.create<vector_ext::YieldOp>(loc, defsWithExternalUses);
    truePredTerminator->erase();
  }
}

/// Utility that predicates a tiled loop with a vector.predicate operation. The
/// vectorization factor used for predication is assumed to be the step of the
/// tiled loop.
LogicalResult mlir::vector_ext::predicateTiledLoop(TiledLoopOp loopOp) {
  if (loopOp.lowerBound().size() > 1)
    // TODO: Support multi-dim tiled loops.
    return failure();

  // Retrieve vectorization factor from the step of the tiled loop.
  Value step = loopOp.step()[0];
  auto maybeVecFactor = getConstantIntValue(step);
  if (!maybeVecFactor)
    return failure();
  int64_t vecFactor = *maybeVecFactor;

  createVectorPredicateOp(loopOp, vecFactor);

  // TODO: Canonicalize affine.min ops feeding extract/insert slices guarded by
  // vector.predicate.

  return success();
}
