//===- StagedPatternRewriteDriver.h - Staged Pattern Driver -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares methods for applying a set of patterns in a staged
// fashion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_STAGEDPATTERNREWRITEDRIVER_H_
#define MLIR_TRANSFORMS_STAGEDPATTERNREWRITEDRIVER_H_

#include "mlir/Rewrite/FrozenRewritePatternSet.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// applyStagedPatterns
//===----------------------------------------------------------------------===//

/// Applies the specified rewrite patterns from `stage1Patterns` on `rootOp`,
/// and its descendants, one pattern patternapplication at a time, interleaved
/// with:
///   a. `stage2Patterns` applied greedily on the parent FuncOp, followed by;
///   b. `stage3Lambda` that just applies some global transformations or passes
///      to the parent FuncOp.
LogicalResult applyStagedPatterns(
    ArrayRef<Operation *> roots, const FrozenRewritePatternSet &stage1Patterns,
    const FrozenRewritePatternSet &stage2Patterns = FrozenRewritePatternSet(),
    function_ref<LogicalResult(FuncOp)> stage3Lambda = {});

} // namespace mlir

#endif // MLIR_TRANSFORMS_STAGEDPATTERNREWRITEDRIVER_H_
