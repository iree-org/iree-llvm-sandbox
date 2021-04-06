//===- Transforms.h - Custom Transforms: TileGeneric+Bufferize --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_EXPERIMENTAL_RUNNERS_TRANSFORMS_H_
#define IREE_EXPERIMENTAL_RUNNERS_TRANSFORMS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"

namespace mlir {
namespace linalg {

/// Options for tile-n-distribute via `linalg.tiled_loop`.
struct TileAndDistributeOptions {
  LinalgTilingOptions tilingOptions;
};

void populateTileAndDistributePattern(OwningRewritePatternList &patterns,
                                      const TileAndDistributeOptions &opts,
                                      const LinalgTransformationFilter &filter);

}  // namespace linalg
}  // namespace mlir

#endif  // IREE_EXPERIMENTAL_RUNNERS_TRANSFORMS_H_
