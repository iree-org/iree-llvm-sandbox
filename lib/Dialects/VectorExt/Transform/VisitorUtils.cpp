// Note: This file is mostly a copy-paste of
// tensorflow/compiler/mlir/tensorflow/utils/visitor_util.h. We copied it here
// for experimentation purposes within the VectorExt dialect. Please, do not
// create a dependency with it outside of the VectorExt dialect since it will go
// away.

//===- VisitorUtils.cpp -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "VisitorUtils.h"
#include "mlir/IR/Operation.h"

namespace mlir {

WalkStage::WalkStage(mlir::Operation *op)
    : numRegions(op->getNumRegions()), nextRegion(0) {}

namespace detail {

/// Walk all of the operations nested under and including the given operations.
void walkOperations(mlir::Operation *op, VoidCallback callback) {
  WalkStage stage(op);

  for (auto &region : op->getRegions()) {
    // Invoke callback on the parent op before visiting each child region.
    callback(op, stage);
    stage.advance();

    for (auto &block : region)
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        walkOperations(&nestedOp, callback);
  }

  // Invoke callback after all regions have been visited.
  callback(op, stage);
}

/// Walk all of the operations nested under and including the given operations.
/// This methods walks operations until an interrupt signal is received.
mlir::WalkResult walkOperations(mlir::Operation *op,
                                InterruptCallback callback) {
  WalkStage stage(op);

  for (auto &region : op->getRegions()) {
    // Invoke callback on the parent op before visiting each child region.
    if (callback(op, stage).wasInterrupted())
      return mlir::WalkResult::interrupt();

    stage.advance();

    for (auto &block : region) {
      // Early increment here in the case where the operation is erased.
      for (auto &nestedOp : llvm::make_early_inc_range(block))
        if (walkOperations(&nestedOp, callback).wasInterrupted())
          return mlir::WalkResult::interrupt();
    }
  }
  return callback(op, stage);
}

} // namespace detail
} // namespace mlir
