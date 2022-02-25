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
#include "Dialects/VectorExt/VectorExtOps.h"

namespace mlir {
class RewritePatternSet;
namespace vector {
class TransferWriteOp;
}
namespace vector_ext {

/// Collect patterns to propagate warp distribution.
void populatePropagateVectorDistributionPatterns(RewritePatternSet &pattern);

void populateWarpSingleLaneOpToScfForPattern(RewritePatternSet &patterns);

using DistributionMapFn = std::function<AffineMap(vector::TransferWriteOp)>;

/// Distribute transfer_write ops based on the affine map returned by
/// `distributionMapFn`.
/// Example:
/// ```
/// %0 = vector_ext.warp_execute_on_lane_0(%id){
///   ...
///   vector.transfer_write %v, %A[%c0] : vector<32xf32>, memref<128xf32>
///   vector_ext.yield
/// }
/// ```
/// To
/// ```
/// %r:3 = vector_ext.warp_execute_on_lane_0(%id) -> (vector<1xf32>) {
///   ...
///   vector_ext.yield %v : vector<32xf32>
/// }
/// vector.transfer_write %v, %A[%id] : vector<1xf32>, memref<128xf32>
void populateDistributeTransferWriteOpPatterns(
    RewritePatternSet &patterns, DistributionMapFn distributionMapFn);

/// Move scalar operations with no dependency on warp op outside of the region.
void moveScalarUniformCode(WarpSingleLaneOp op);

} // namespace vector_ext
} // namespace mlir

#endif // DIALECT_VECTOREXT_VECTORWARPUTILS_H_
