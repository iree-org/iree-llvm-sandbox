//===- Transforms.h - Custom Transforms: TileGeneric+Bufferize --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_TRANSFORMS_TRANSFORMS_H_
#define IREE_LLVM_SANDBOX_TRANSFORMS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace linalg {

void populateTileAndFusePattern(RewritePatternSet &patterns,
                                const LinalgTilingOptions &opts,
                                const LinalgTransformationFilter &filter);

void populateFuseFillIntoReductionPatterns(RewritePatternSet &patterns);

void populateTiledLoopsToSCF(RewritePatternSet &patterns);

void populateTiledLoopsToGPUPatterns(RewritePatternSet &patterns,
                                     ArrayRef<int64_t> numWorkgroups);

void populateDistributeTiledLoopPattern(
    RewritePatternSet &patterns,
    const LinalgLoopDistributionOptions &opts,
    const LinalgTransformationFilter &filter);

void populateTiledLoopToAsyncPatterns(RewritePatternSet &patterns);

}  // namespace linalg
}  // namespace mlir

#endif // IREE_LLVM_SANDBOX_TRANSFORMS_TRANSFORMS_H_
