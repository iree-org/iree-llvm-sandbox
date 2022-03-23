//===-- VectorMaskingUtils.h - Utilities for vector masking -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_
#define DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace func {
class FuncOp;
} // namespace func
class LogicalResult;
class Region;
class OpBuilder;
class Operation;
class Value;
class ValueRange;
class WalkStage;

namespace vector_ext {
class PredicateOp;

/// Introduce a vector.predicate op that encloses all the operations in
/// `regionToPredicate` (except its terminator). `createPredicate` is used to
/// create the operations that generate the predicate used by vector.predicate.
/// These operations will be inserted at the beginning of `regionToPredicate`.
/// The new vector.predicate will be inserted right after the operations
/// generating the predicate.
llvm::Optional<PredicateOp>
predicateOp(OpBuilder &builder, Operation *op, Region *regionToPredicate,
            llvm::function_ref<Value(OpBuilder &)> createPredicateMask,
            ValueRange indexes, llvm::Optional<Value> maybeIncomingMask);

/// Function signature of a masking strategy for generic operations.
using GenericOpMaskingStrategy =
    llvm::function_ref<void(OpBuilder &, Operation *, Value, const WalkStage &,
                            llvm::SmallVectorImpl<Operation *> &)>;

/// Traverse `op` and apply masking on all the vector.predicate ops and their
/// enclosing operations using the strategy `maskGenericOp` to mask generic
/// operations other than the vector.predicate.
LogicalResult maskVectorPredicateOps(OpBuilder &builder, Operation *op,
                                     GenericOpMaskingStrategy maskGenericOp);

//===----------------------------------------------------------------------===//
// Vector Masking Strategies.
//===----------------------------------------------------------------------===//

/// Masking strategy that only masks vector transfer operations and operations
/// with side effects. Non-side-effecting ops are left unmasked.
void maskGenericOpWithSideEffects(
    OpBuilder &builder, Operation *op, Value activeMask, const WalkStage &stage,
    llvm::SmallVectorImpl<Operation *> &erasedOps);

// TODO: Implement full masking strategy.

} // namespace vector_ext
} // namespace mlir

#endif // DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_
