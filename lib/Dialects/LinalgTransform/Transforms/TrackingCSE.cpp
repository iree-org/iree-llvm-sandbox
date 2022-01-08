//===-- TrackingCSE.cpp - Common subexpr elimination keeping track of ops -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/TrackingCSE.h"
#include "TrackingListener.h"
#include "Transforms/Listener.h"
#include "Transforms/ListenerCSE.h"

using namespace mlir;

void mlir::eliminateCommonSubexpressionsWithTrackedOps(
    Operation *root, TransformOpMapping &trackedOps,
    DominanceInfo *domInfo) {
  linalg::TrackingListener listener(trackedOps);
  (void)eliminateCommonSubexpressions(root, domInfo, &listener);
}
