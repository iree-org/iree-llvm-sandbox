//===-- TransformOpInterface.h - Interface for transform ops ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LINALG_TRANSFORM_TRANSFORM_OP_INTERFACE_H
#define MLIR_DIALECT_LINALG_TRANSFORM_TRANSFORM_OP_INTERFACE_H

#include "Dialects/LinalgTransform/TransformOpMapping.h"
#include "Transforms/Functional.h"
#include <mlir/IR/OpDefinition.h>
#include <type_traits>

namespace mlir {
namespace linalg {
namespace transform {

class TransformOpInterface;

/// The state maintained across applications of various ops implementing the
/// TransformOpInterface. The operations implementing this interface and the
/// surrounding structure are referred to as transform IR. The operations to
/// which transformations apply are referred to as payload IR. The state thus
/// contains the mapping between values defined transform IR ops and payload IR
/// ops. It assumes that each value in the transform IR can be used at most once
/// (since transformations are likely to change the payload IR ops the value
/// corresponds to). Checks that transform IR values correspond to disjoint sets
/// of payload IR ops throughout the transformation.
class TransformState {
public:
  /// Creates a state for the transformation rooted at the given op.
  explicit TransformState(Operation *root);

  /// Returns the op at which the transformation state is rooted. This is
  /// typically helpful for transformations that apply globally.
  Operation *getTopLevel() const;

  /// Returns the list of ops that the given transform IR value corresponds to.
  /// This is helpful for transformations that apply to a particular handle.
  ArrayRef<Operation *> getPayloadOps(Value value) const;

  /// Applies the transformation specified by the given transform op and updates
  /// the state accordingly.
  LogicalResult applyTransform(TransformOpInterface transform);

  // FIXME: we shouldn't be exposing this as it allows for modification of the
  // internal mapping state.
  TransformOpMapping &getMapping() { return operations; }

private:
  /// Identifier for storing top-level value in the `operations` mapping.
  constexpr const static Value kTopLevelValue = Value();

  /// Sets the payload IR ops associated with the given transform IR value.
  /// Fails if this would result in multiple transform IR values with uses
  /// corresponding to the same payload IR ops.
  LogicalResult setPayloadOps(Value value, ArrayRef<Operation *> targets);

  /// Forgets the payload IR ops associated with the given transform IR value.
  void removePayloadOps(Value value);

  /// The mapping between payload IR values and transform IR ops.
  TransformOpMapping operations;
};

/// Local mapping between values defined by a specific op implementing the
/// TransformOpInterface and the payload IR ops they correspond to.
class TransformResults {
  friend class TransformState;

public:
  /// Indicates that the result of the transform IR op at the given position
  /// corresponds to the given list of payload IR ops. Each result must be set
  /// by the transformation exactly once.
  void set(OpResult value, ArrayRef<Operation *> ops);

private:
  /// Creates an instance of TransformResults that expects mappings for
  /// `numSegments` values.
  explicit TransformResults(unsigned numSegments);

  /// Gets the list of operations associated with the result at the given
  /// position.
  ArrayRef<Operation *> get(unsigned position) const;

  /// Storage for pointers to payload IR ops that are associated with results of
  /// a transform IR op. `segments` contains as many entries as the transform IR
  /// op has results. Each entry is a reference to a contiguous segment in
  /// the `operations` list that contains the pointers to operations. This
  /// allows for operations to be stored contiguously without nested vectors and
  /// for different segments to be set in any order.
  SmallVector<ArrayRef<Operation *>, 2> segments;
  SmallVector<Operation *> operations;
};

namespace detail {
/// Appends `result` to the vector assuming it corresponds to the success state
/// in `FailureOr<convertible-to-Operation*>`. If `result` is just a
/// `LogicalResult`, does nothing.
template <typename Ty>
std::enable_if_t<std::is_same<Ty, LogicalResult>::value>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {}

template <typename Ty>
std::enable_if_t<!std::is_same<Ty, LogicalResult>::value>
appendTransformResultToVector(Ty result,
                              SmallVectorImpl<Operation *> &results) {
  static_assert(
      std::is_convertible<typename Ty::value_type, Operation *>::value,
      "Expected transform function to return operations");
  results.push_back(*result);
}
} // namespace detail

/// Applies a one-to-one transform to each of the given targets. Puts the
/// results of transforms, if any, in `results` in the same order. Fails if any
/// of the application fails. Individual transforms must be callable with
/// one of the following signatures:
///   - FailureOr<convertible-to-Operation*>(OpTy)
///   - LogicalResult(OpTy)
/// where OpTy is either
///   - Operation *, in which case the transform is always applied;
///   - a concrete Op class, in which case a check is performed whether
///   `targets` contains operations of the same class and a failure is reported
///   if it does not.
template <typename FnTy>
LogicalResult applyTransformToEach(ArrayRef<Operation *> targets,
                                   SmallVectorImpl<Operation *> &results,
                                   FnTy transform) {

  using TransformOpType =
      typename llvm::function_traits<FnTy>::template arg_t<0>;
  static_assert(std::is_convertible<TransformOpType, Operation *>::value,
                "Expected transform function to take an operation");
  for (Operation *target : targets) {
    auto specificOp =
        functional::detail::IsaOr<TransformOpType>::dyn_cast(target);
    if (!specificOp)
      return failure();

    auto result = transform(specificOp);
    if (failed(result))
      return failure();

    detail::appendTransformResultToVector(result, results);
  }
  return success();
}

/// Trait implementing the TransformOpInterface for operations applying a
/// transformation to a single operation handle and producing a single operation
/// handle. The op must implement a method with one of the following signatures:
///   - FailureOr<convertible-to-Operation*> applyToOne(OpTy)
///   - LogicalResult applyToOne(OpTy)
/// to perform a transformation that is applied in turn to all payload IR
/// operations that correspond to the handle of the transform IR operation.
/// In the functions above, OpTy is either Operation * or a concrete payload IR
/// Op class that the transformation is applied to (NOT the class of the
/// transform IR op).
template <typename OpTy>
class TargetableSingleOperandOpTrait
    : public OpTrait::TraitBase<OpTy, TargetableSingleOperandOpTrait> {
public:
  /// Applies the transformation to each op from the only target and sets the
  /// only result to correspond to the list of individual results.
  LogicalResult apply(TransformResults &transformResults,
                      TransformState &state) {
    using TransformOpType = typename llvm::function_traits<
        decltype(&OpTy::applyToOne)>::template arg_t<0>;
    ArrayRef<Operation *> targets =
        state.getPayloadOps(this->getOperation()->getOperand(0));
    SmallVector<Operation *> results;
    if (failed(applyTransformToEach(
            targets, results, [&](TransformOpType specificOp) {
              return static_cast<OpTy *>(this)->applyToOne(specificOp);
            })))
      return failure();
    if (OpTy::template hasTrait<OpTrait::OneResult>()) {
      transformResults.set(
          this->getOperation()->getResult(0).template cast<OpResult>(),
          results);
    }
    return success();
  }

  /// Verifies that the op satisfies the requirements for this trait.
  static LogicalResult verifyTrait(Operation *) {
    static_assert(OpTy::template hasTrait<OpTrait::OneOperand>(),
                  "expected single-operand op");
    static_assert(OpTy::template hasTrait<OpTrait::OneResult>() ||
                      OpTy::template hasTrait<OpTrait::ZeroResult>(),
                  "expected zero- or single-result op");
    return success();
  }
};

} // namespace transform
} // namespace linalg
} // namespace mlir

#include "Dialects/LinalgTransform/TransformOpInterface.h.inc"

#endif // MLIR_DIALECT_LINALG_TRANSFORM_TRANSFORM_OP_INTERFACE_H
