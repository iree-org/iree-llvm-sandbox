//===-- TransformOpInterface.cpp - Interface for transform ops ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/TransformOpInterface.h"

#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
using namespace mlir::linalg;

//===----------------------------------------------------------------------===//
// TransformState
//===----------------------------------------------------------------------===//

constexpr const Value transform::TransformState::kTopLevelValue;

transform::TransformState::TransformState(Operation *root) {
  operations[kTopLevelValue].push_back(root);
}

Operation *transform::TransformState::getTopLevel() const {
  return operations.lookup(kTopLevelValue).front();
}

ArrayRef<Operation *>
transform::TransformState::getPayloadOps(Value value) const {
  auto iter = operations.find(value);
  assert(iter != operations.end() && "unknown handle");
  return iter->getSecond();
}

LogicalResult
transform::TransformState::setPayloadOps(Value value,
                                         ArrayRef<Operation *> targets) {
  assert(value != kTopLevelValue &&
         "attempting to reset the transformation root");

  if (value.use_empty())
    return success();

  SmallVector<Operation *> storedTargets(targets.begin(), targets.end());
  bool inserted = operations.insert({value, std::move(storedTargets)}).second;
  assert(inserted && "value is already associated with another list");
  (void)inserted;

  const SmallVector<Operation *> &currentOperationList =
      operations.lookup(value);
  llvm::SmallPtrSet<Operation *, 4> currentOperationSet(
      currentOperationList.begin(), currentOperationList.end());
  for (const auto &kvp : operations) {
    if (kvp.getFirst() == value)
      continue;
    for (Operation *trackedOp : kvp.getSecond()) {
      if (currentOperationSet.contains(trackedOp)) {
        InFlightDiagnostic diag = trackedOp->emitError()
                                  << "operation tracked by two handles";
        diag.attachNote(value.getLoc()) << "handle";
        diag.attachNote(kvp.getFirst().getLoc()) << "handle";
        return diag;
      }
    }
  }

  return success();
}

void transform::TransformState::removePayloadOps(Value value) {
  operations.erase(value);
}

LogicalResult
transform::TransformState::applyTransform(TransformOpInterface transform) {
  transform::TransformResults results(transform->getNumResults());
  if (failed(transform.apply(results, *this)))
    return failure();

  for (auto en : llvm::enumerate(transform->getResults()))
    if (failed(setPayloadOps(en.value(), results.get(en.index()))))
      return failure();

  for (Value target : transform->getOperands())
    removePayloadOps(target);
  return success();
}

//===----------------------------------------------------------------------===//
// TransformResults
//===----------------------------------------------------------------------===//

transform::TransformResults::TransformResults(unsigned numSegments) {
  segments.resize(numSegments,
                  ArrayRef<Operation *>(nullptr, static_cast<size_t>(0)));
}

void transform::TransformResults::set(OpResult value,
                                      ArrayRef<Operation *> ops) {
  unsigned position = value.getResultNumber();
  assert(position < segments.size() &&
         "setting results for a non-existent handle");
  assert(segments[position].data() == nullptr && "results already set");
  unsigned start = operations.size();
  llvm::append_range(operations, ops);
  segments[position] = makeArrayRef(operations).drop_front(start);
}

ArrayRef<Operation *>
transform::TransformResults::get(unsigned position) const {
  assert(position < segments.size() &&
         "querying results for a non-existent handle");
  assert(segments[position].data() != nullptr && "querying unset results");
  return segments[position];
}

//===----------------------------------------------------------------------===//
// Generated interface implementation.
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/TransformOpInterface.cpp.inc"
