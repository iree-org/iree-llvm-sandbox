//===-- VectorMaskingUtils.h - Utilities for vector masking -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_
#define DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_

namespace mlir {
class LogicalResult;

namespace linalg {
class TiledLoopOp;
}

namespace vector_ext {

/// Utility that predicates a tiled loop with a vector.predicate operation. The
/// vectorization factor used for predication is assumed to be the step of the
/// tiled loop.
LogicalResult predicateTiledLoop(linalg::TiledLoopOp loopOp);

} // namespace vector_ext
} // namespace mlir

#endif // DIALECT_VECTOREXT_VECTORMASKINGUTILS_H_
