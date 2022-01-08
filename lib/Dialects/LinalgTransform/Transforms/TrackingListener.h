//===-- TrackingListener.h - Common listener for tracking passes ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_TRACKINGLISTENER_H
#define IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_TRACKINGLISTENER_H

#include "Transforms/Listener.h"
#include "Dialects/LinalgTransform/TransformOpMapping.h"

namespace mlir {
namespace linalg {
/// A tracking listener using to perform CSE and canonicalization passes while
/// tracking certain linalg operation handles live in a linalg transform
/// interpreter.
class TrackingListener : public RewriteListener {
public:
  TrackingListener(TransformOpMapping &trackedOperations);

  /// When a tracked linalg operation is replaced, try to find a single linalg
  /// op responsible for the replacement values and substitute the handle of the
  /// replaced op for this op.
  void notifyOperationReplaced(Operation *op, ValueRange newValues) override;

  /// When a tracked operation is removed (due to CSE or canonicalization), then
  /// any further transformations on the op are redundant. Remove it from the
  /// tracked operation list.
  void notifyOperationRemoved(Operation *op) override;

private:
  /// A reference to the tracked operations in the interpreter.
  TransformOpMapping &trackedOperations;
  /// A map from a tracked operation (LinalgOp cannot be used as a key) to its
  /// key in the map.
  DenseMap<Operation *, Value> trackedOperationKeys;
};
} // namespace linalg
} // namespace mlir

#endif // IREE_LLVM_SANDBOX_DIALECTS_LINALGTRANSFORM_TRANSFORMS_TRACKINGLISTENER_H
