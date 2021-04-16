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
  using edsc::op::operator-;
  return Range{lb, ub - lb, step};
}

class DistributePass : public PassWrapper<DistributePass, FunctionPass> {
 public:
  DistributePass() : PassWrapper() {}

  DistributePass(LinalgLoopDistributionOptions options)
      : PassWrapper(), options_(std::move(options)) {}

  void runOnFunction() override {
    FuncOp func = getFunction();
    Location loc = func.getLoc();

    OpBuilder builder(func.getBody());
    edsc::ScopedContext scope(builder, loc);
    func.walk([&](TiledLoopOp op) {
      SmallVector<Range, 2> parallelLoopRanges;
      parallelLoopRanges.reserve(op.getNumLoops());

      // Get ranges for parallel loop dimensions.
      unsigned numLoops = op.getNumLoops();
      unsigned numLoopsToDistribute = std::min(
          numLoops, static_cast<unsigned>(options_.distributionMethod.size()));
      if (numLoopsToDistribute == 0) return;

      OpBuilder& b = edsc::ScopedContext::getBuilderRef();
      b.setInsertionPoint(op);
      for (int i = 0; i < numLoopsToDistribute; ++i) {
        if (!isParallelIteratorType(op.iterator_types()[i])) {
          op.emitOpError("only support for parallel loops is implemented");
          return;
        }
        parallelLoopRanges.push_back(ConstructRange(
            loc, b, op.lowerBound()[i], op.upperBound()[i], op.step()[i]));
      }

      // Get processor info.
      SmallVector<ProcInfo, 2> procInfos =
          options_.procInfo(b, loc, parallelLoopRanges);

      // Update bounds and steps.
      SmallVector<Value, 2> newLowerBounds = op.lowerBound();
      SmallVector<Value, 2> newUpperBounds = op.upperBound();
      SmallVector<Value, 2> newSteps = op.step();
      for (auto& en : llvm::enumerate(procInfos)) {
        size_t index = en.index();
        updateBoundsForCyclicDistribution(
            b, loc, procInfos[index].procId, procInfos[index].nprocs,
            newLowerBounds[index], newUpperBounds[index], newSteps[index]);
      }
      op.setLowerBounds(newLowerBounds);
      op.setUpperBounds(newUpperBounds);
      op.setSteps(newSteps);
    });
  }

  LinalgLoopDistributionOptions options_;
};

static SmallVector<linalg::ProcInfo, 2> getGpuProcIds(
    OpBuilder& b, Location loc, ArrayRef<Range> parallelLoopRanges) {
  if (parallelLoopRanges.size() != 2)
    llvm_unreachable("expected two parallel loops");
  Type indexType = b.getIndexType();
  return {{b.create<gpu::BlockIdOp>(loc, indexType, b.getStringAttr("y")),
           b.create<gpu::GridDimOp>(loc, indexType, b.getStringAttr("y"))},
          {b.create<gpu::BlockIdOp>(loc, indexType, b.getStringAttr("x")),
           b.create<gpu::GridDimOp>(loc, indexType, b.getStringAttr("x"))}};
}

class TestDistributePass : public DistributePass {
 public:
  TestDistributePass()
      : DistributePass(LinalgLoopDistributionOptions{
            getGpuProcIds,
            {DistributionMethod::Cyclic, DistributionMethod::Cyclic}}) {}
  TestDistributePass(const TestDistributePass&) = default;

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<gpu::GPUDialect>();
  }
};

}  // namespace

namespace mlir {
namespace linalg {

std::unique_ptr<OperationPass<FuncOp>> createDistributePass(
    LinalgLoopDistributionOptions options) {
  return std::make_unique<DistributePass>(options);
}

void registerLinalgDistributePass() {
  PassRegistration<TestDistributePass> testDistributePass(
      "test-linalg-distribute-tiled-loop",
      "Distribute Linalg Tiled Loop Pass.");
}

}  // namespace linalg
}  // namespace mlir
