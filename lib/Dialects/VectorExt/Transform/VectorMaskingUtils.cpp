//===- VectorMaskingUtils.cpp - Utilities for vector masking --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorMaskingUtils.h"
#include "Dialects/VectorExt/VectorExtOps.h"
#include "VisitorUtils.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/Support/LogicalResult.h"
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
getDefsWithUsesOutside(llvm::iterator_range<Block::iterator> opRange,
                       SmallVectorImpl<Value> &defsWithUsesOutside,
                       SmallVectorImpl<Type> &defTypes) {
  SmallPtrSet<Operation *, 8> rangeSet;
  for (Operation &op : opRange)
    rangeSet.insert(&op);

  for (Operation &op : opRange)
    for (Value res : op.getResults())
      for (Operation *user : res.getUsers())
        if (rangeSet.count(user) == 0) {
          defsWithUsesOutside.push_back(res);
          defTypes.push_back(res.getType());
          break;
        }
}

/// Introduce a vector.predicate op that encloses all the operations in
/// `regionToPredicate` (except its terminator). `createPredicate` is used to
/// create the operations that generate the predicate used by vector.predicate.
/// These operations will be inserted at the beginning of `regionToPredicate`.
/// The new vector.predicate will be inserted right after the operations
/// generating the predicate.
Optional<PredicateOp> mlir::vector_ext::predicateOp(
    OpBuilder &builder, Operation *op, Region *regionToPredicate,
    function_ref<Value(OpBuilder &)> createPredicate) {
  // TODO: Support multi-block regions.
  if (!regionToPredicate->hasOneBlock())
    return llvm::None;

  // Compute the range of operations that will be moved within vector.predicate
  // and the definitions within the range with users outside the range.
  Block &blockToPredicate = regionToPredicate->front();
  auto opsToMove = llvm::make_range(blockToPredicate.begin(),
                                    Block::iterator(blockToPredicate.back()));

  SmallVector<Value, 8> defsWithUsesOutside;
  SmallVector<Type, 8> defTypes;
  getDefsWithUsesOutside(opsToMove, defsWithUsesOutside, defTypes);

  // Set the builder insertion point to the beginning of the loop body to insert
  // the predicate computation and the vector.predicate.
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&blockToPredicate);
  Value predicate = createPredicate(builder);

  // Generate the vector.predicate operation and move 'opsToMove' within its
  // truePredicateRegion. We have to rewire the def-use chain for those
  // definitions within the range that have external uses. Those uses are
  // rewired to the results of the vector.predicate.
  Location loc = op->getLoc();
  auto vecPredOp = builder.create<PredicateOp>(loc, defTypes, predicate);
  assert(defsWithUsesOutside.size() == vecPredOp.getNumResults() &&
         "Expected same size");
  for (auto &en : llvm::enumerate(defsWithUsesOutside))
    en.value().replaceAllUsesWith(vecPredOp.getResult(en.index()));

  Operation *truePredTerminator =
      &vecPredOp.truePredicateRegion().front().back();
  moveOperationsBefore(opsToMove, truePredTerminator);

  // The existing terminator of TruePredicateRegion doesn't yield any value.
  // Replace it with a new terminator that returns the definitions with uses
  // outside that we just moved.
  if (vecPredOp.getNumResults() > 0) {
    builder.setInsertionPoint(truePredTerminator);
    builder.create<vector_ext::YieldOp>(loc, defsWithUsesOutside);
    truePredTerminator->erase();
  }

  return vecPredOp;
}

/// Utility that predicates the body a tiled loop with a vector.predicate
/// operation. The vectorization factor used for predication is assumed to be
/// the step of the tiled loop.
LogicalResult mlir::vector_ext::predicateTiledLoop(OpBuilder &builder,
                                                   TiledLoopOp loopOp) {
  if (loopOp.lowerBound().size() > 1)
    // TODO: Support multi-dim tiled loops.
    return failure();

  // Retrieve vectorization factor from the step of the tiled loop.
  Value step = loopOp.step()[0];
  auto maybeVecFactor = getConstantIntValue(step);
  if (!maybeVecFactor)
    return failure();
  int64_t vecFactor = *maybeVecFactor;

  auto createPredicate = [&](OpBuilder &builder) -> Value {
    // Generate the predicate to be used by the vector.predicate operation:
    //   %min = affine.min affine_map<(d0)[s0] -> (vecFactor, -d0 +
    //   s0)>(%iv)[%ub] %vec_pred = vector.create_mask %min : vector<8xi1>
    // TODO: This is probably not the most efficient way to generate the vector
    // predicate. Consider using the vector IV.
    AffineExpr i, j;
    bindDims(loopOp.getContext(), i, j);
    SmallVector<AffineMap, 4> maps = AffineMap::inferFromExprList(
        ExprList{{builder.getAffineConstantExpr(vecFactor), i - j}});

    Location loc = loopOp.getLoc();
    auto minOp = builder.create<AffineMinOp>(
        loc, loopOp.step()[0].getType(), maps[0],
        ValueRange{loopOp.upperBound()[0], loopOp.getInductionVars()[0]});

    auto maskType = VectorType::get({vecFactor}, builder.getI1Type());
    return builder.create<CreateMaskOp>(loc, maskType, ValueRange{minOp});
  };

  auto mayPredicate =
      predicateOp(builder, loopOp, &loopOp.getLoopBody(), createPredicate);
  if (!mayPredicate)
    return failure();

  // TODO: Canonicalize affine.min ops feeding extract/insert slices guarded by
  // vector.predicate.

  return success();
}

/// Materialize masking on a vector.predicate op by generating the masks to
/// apply to its enclosing operations and erasing the vector.predicate op after
/// inlining its enclosing operations into its parent region.
static void maskPredicateOp(OpBuilder &builder, PredicateOp predOp,
                             SmallVectorImpl<Value> &activeMasks,
                             const WalkStage &stage) {
  // Actions before visiting the TruePredicateRegion: Update active mask to be
  // the predicate mask.
  if (stage.isBeforeAllRegions()) {
    activeMasks.push_back(predOp.predicate());
    return;
  }

  // Actions after visiting the TruePredicateRegion: Pop the active mask for the
  // TruePredicateRegion.
  // if (stage.isAfterRegion(0)) {
  //   activeMasks.pop_back();
  //   return;
  // }

  // TODO: Generate complementary mask for FalsePredicateRegion.
  // if (stage.isBeforeRegion(1)) {
  //   return;
  // }

  // Actions after all the regions: Move operations outside the vector.predicate
  // regions and remove vector.predicate op.
  if (stage.isAfterAllRegions()) {
    // TODO: Move this to isAfterRegion(0);
    activeMasks.pop_back();

    // Inline truePredicateRegion into parent op (except its terminator).
    // TODO: Move this to builder.inlineRegionBefore?
    auto &blocksToMove = predOp.truePredicateRegion().getBlocks();
    assert(blocksToMove.size() == 1 && "Expected only one block");
    assert(predOp.getResults().empty() &&
           "TODO: Support vector.predicate with results");
    auto &opsToMove = blocksToMove.front().getOperations();
    moveOperationsBefore(
        llvm::make_range(opsToMove.begin(), std::prev(opsToMove.end())),
        predOp);
    predOp.erase();
    return;
  }
}

/// Traverse `op` and apply masking on all the vector.predicate ops and their
/// enclosing operations using the strategy `maskGenericOp` to mask generic
/// operations other than the vector.predicate.
// TODO: Add an initMask param to pass a mask that is not all-ones?
LogicalResult mlir::vector_ext::maskVectorPredicateOps(
    OpBuilder &builder, Operation *op, GenericOpMaskingStrategy maskGenericOp) {
  OpBuilder::InsertionGuard guard(builder);

  // Initialize stack of active masks. An empty stack means that the active mask
  // is all-ones (masking is not required).
  SmallVector<Value, 8> activeMasks;
  activeMasks.push_back(Value());

  genericWalk(op, [&](Operation *op, const WalkStage &stage) {
    if (auto predOp = dyn_cast<PredicateOp>(op))
      maskPredicateOp(builder, predOp, activeMasks, stage);
    else
      maskGenericOp(builder, op, activeMasks.back(), stage);
  });

  return success();
}

/// Masking strategy that only masks vector transfer operations and operations
/// with side effects. Non-side-effecting ops are left unmasked.
void mlir::vector_ext::maskGenericOpMinimalMaskingStrategy(
    OpBuilder &builder, Operation *op, Value activeMask,
    const WalkStage &stage) {
  // Nothing to do. All-ones mask to apply.
  if (!activeMask)
    return;

  if (stage.isBeforeAllRegions()) {
    // Mask vector.transfer_read.
    if (auto xferReadOp = dyn_cast<TransferReadOp>(op)) {
      builder.setInsertionPoint(op);
      auto newXferOp = builder.create<TransferReadOp>(
          op->getLoc(), xferReadOp.getType(), xferReadOp.source(),
          xferReadOp.indices(), xferReadOp.permutation_map(),
          xferReadOp.padding(), activeMask, xferReadOp.in_boundsAttr());
      xferReadOp.replaceAllUsesWith(newXferOp.getResult());
      xferReadOp.erase();
      return;
    }

    // Mask vector.transfer_read.
    if (auto xferWriteOp = dyn_cast<TransferWriteOp>(op)) {
      builder.setInsertionPoint(op);
      builder.create<TransferWriteOp>(
          op->getLoc(), xferWriteOp.vector(), xferWriteOp.source(),
          xferWriteOp.indices(), xferWriteOp.permutation_mapAttr(), activeMask,
          xferWriteOp.in_boundsAttr());
      xferWriteOp.erase();
      return;
    }
  }
}
