//===- LinalgDistributePass.cpp - Distribute to linalg.tile ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic to distribute `linalg.tiled_loop`.
//
//===----------------------------------------------------------------------===//

#include "Transforms.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "distribute"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;
using namespace mlir::linalg;

namespace {

Range ConstructRange(Location loc, OpBuilder& b, Value lb, Value ub,
                     Value step) {
  AffineExpr d0, d1;
  bindDims(b.getContext(), d0, d1);
  return Range{lb, makeComposedAffineApply(b, loc, d0 - d1, {ub, lb}), step};
}

struct DistributeTiledLoopPattern
    : public OpRewritePattern<linalg::TiledLoopOp> {
  DistributeTiledLoopPattern(MLIRContext* context,
                             LinalgLoopDistributionOptions options,
                             LinalgTransformationFilter filter)
      : OpRewritePattern<linalg::TiledLoopOp>(context),
        options(options),
        filter(filter) {}
  LogicalResult matchAndRewrite(linalg::TiledLoopOp op,
                                PatternRewriter& rewriter) const override {
    if (failed(filter.checkAndNotify(rewriter, op))) return failure();
    Location loc = op.getLoc();
    SmallVector<Range, 2> parallelLoopRanges;
    parallelLoopRanges.reserve(op.getNumLoops());

    // Get ranges for parallel loop dimensions.
    unsigned numLoops = op.getNumLoops();
    unsigned numLoopsToDistribute = std::min(
        numLoops, static_cast<unsigned>(options.distributionMethod.size()));
    if (numLoopsToDistribute == 0) return failure();

    edsc::ScopedContext scope(rewriter, loc);
    for (int i = 0; i < numLoopsToDistribute; ++i) {
      if (!isParallelIteratorType(op.iterator_types()[i])) {
        op.emitOpError("only support for parallel loops is implemented");
        return failure();
      }
      parallelLoopRanges.push_back(ConstructRange(
          loc, rewriter, op.lowerBound()[i], op.upperBound()[i], op.step()[i]));
    }

    // Get processor info.
    SmallVector<ProcInfo, 2> procInfos =
        options.procInfo(rewriter, loc, parallelLoopRanges);

    // Update bounds and steps.
    SmallVector<Value, 2> newLowerBounds = op.lowerBound();
    SmallVector<Value, 2> newUpperBounds = op.upperBound();
    SmallVector<Value, 2> newSteps = op.step();
    for (auto& en : llvm::enumerate(procInfos)) {
      size_t index = en.index();
      updateBoundsForCyclicDistribution(
          rewriter, loc, procInfos[index].procId, procInfos[index].nprocs,
          newLowerBounds[index], newUpperBounds[index], newSteps[index]);
    }
    rewriter.updateRootInPlace(op, [&] {
      op.setLowerBounds(newLowerBounds);
      op.setUpperBounds(newUpperBounds);
      op.setSteps(newSteps);
    });
    filter.replaceLinalgTransformationFilter(rewriter, op);
    return success();
  }

 private:
  LinalgLoopDistributionOptions options;
  LinalgTransformationFilter filter;
};

}  // namespace

namespace mlir {
namespace linalg {

void populateDistributeTiledLoopPattern(
    OwningRewritePatternList& patterns,
    const LinalgLoopDistributionOptions& opts,
    const LinalgTransformationFilter& filter) {
  patterns.add<DistributeTiledLoopPattern>(patterns.getContext(), opts, filter);
}

}  // namespace linalg
}  // namespace mlir
