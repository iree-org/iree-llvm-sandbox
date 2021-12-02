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

namespace mlir {
class LogicalResult;
class Region;
class OpBuilder;
class Operation;
class Value;

namespace linalg {
class TiledLoopOp;
}

namespace vector_ext {
class PredicateOp;

/// Introduce a vector.predicate op that encloses all the operations in
/// `regionToPredicate` (except its terminator). `createPredicate` is used to
/// create the operations that generate the predicate used by vector.predicate.
/// These operations will be inserted at the beginning of `regionToPredicate`.
/// The new vector.predicate will be inserted right after the operations
/// generating the predicate.
llvm::Optional<PredicateOp>
predicateOp(Operation *op, Region *regionToPredicate, OpBuilder &builder,
            llvm::function_ref<Value(OpBuilder &)> createPredicate);

/// Utility that predicates the body a tiled loop with a vector.predicate
/// operation. The vectorization factor used for predication is assumed to be
/// the step of the tiled loop.
LogicalResult predicateTiledLoop(linalg::TiledLoopOp loopOp);

} // namespace vector_ext
} // namespace mlir

#endif // DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_
