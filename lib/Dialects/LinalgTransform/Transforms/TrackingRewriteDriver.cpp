//===-- TrackingRewriteDriver.cpp - Pattern rewriter keeping track of ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/TrackingRewriteDriver.h"
#include "TrackingListener.h"
#include "Transforms/ListenerGreedyPatternRewriteDriver.h"

using namespace mlir;

LogicalResult mlir::applyPatternsTrackAndFoldGreedily(
    Operation *root, TransformOpMapping &trackedOperations,
    const FrozenRewritePatternSet &patterns, GreedyRewriteConfig config) {
  linalg::TrackingListener listener(trackedOperations);
  if (failed(applyPatternsAndFoldGreedily(root, patterns, config, &listener)))
    return failure();
  return listener.checkErrorState();
}
