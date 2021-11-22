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

void populateTileAndFusePattern(OwningRewritePatternList &patterns,
                                const LinalgTilingOptions &opts,
                                const LinalgTransformationFilter &filter);

void populateFuseFillIntoReductionPatterns(OwningRewritePatternList &patterns,
                                           const LinalgTilingOptions &opts);

void populateTiledLoopsToSCF(OwningRewritePatternList &patterns);

void populateTiledLoopsToGPUPatterns(OwningRewritePatternList &patterns,
                                     ArrayRef<int64_t> numWorkgroups);

void populateDistributeTiledLoopPattern(
    OwningRewritePatternList &patterns,
    const LinalgLoopDistributionOptions &opts,
    const LinalgTransformationFilter &filter);

void populateTiledLoopToAsyncPatterns(OwningRewritePatternList &patterns);

}  // namespace linalg
}  // namespace mlir

#endif // IREE_LLVM_SANDBOX_TRANSFORMS_TRANSFORMS_H_
