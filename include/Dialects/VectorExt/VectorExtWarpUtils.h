//===---- VectorWarpUtils.h - Utilities for vector warp ops -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECT_VECTOREXT_VECTORWARPUTILS_H_
#define DIALECT_VECTOREXT_VECTORWARPUTILS_H_

#include "llvm/ADT/STLExtras.h"

namespace mlir {
class RewritePatternSet;
namespace vector {
class TransferWriteOp;
}
namespace vector_ext {

/// Collect patterns to propagate warp distribution.
void populatePropagateVectorDistributionPatterns(RewritePatternSet &pattern);

void populateWarpSingleLaneOpToScfForPattern(RewritePatternSet &patterns);

/// Distribute transfer_write ops based on the affine map returs by
/// `distributionMapFn`.
void distributeTransferWrite(
    OpBuilder &builder, WarpSingleLaneOp op,
    std::function<AffineMap(vector::TransferWriteOp)> distributionMapFn);

/// Move scalar operations with no dependency on warp op outside of the region.
void moveScalarUniformCode(WarpSingleLaneOp op);

} // namespace vector_ext
} // namespace mlir

#endif // DIALECT_VECTOREXT_VECTORWARPUTILS_H_
