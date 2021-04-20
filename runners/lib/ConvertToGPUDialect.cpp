// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include <cstdint>

#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;  // NOLINT

namespace {

static constexpr int numWorkgroupDim = 3;

struct TiledLoopToGPUPattern : public OpRewritePattern<linalg::TiledLoopOp> {
  TiledLoopToGPUPattern(MLIRContext* context, ArrayRef<int64_t> numWorkgroups)
      : OpRewritePattern<linalg::TiledLoopOp>(context),
        numWorkgroups(numWorkgroups.begin(), numWorkgroups.end()) {}
  LogicalResult matchAndRewrite(linalg::TiledLoopOp tiledLoopOp,
                                PatternRewriter& rewriter) const override {
    // Assume the first level TiledLoop op maps to GPU dispatch. In the future
    // there may be some annotations to make it explicit.
    if (tiledLoopOp->getParentOfType<linalg::TiledLoopOp>() ||
        tiledLoopOp->getParentOfType<gpu::LaunchOp>())
      return failure();
    Location loc = tiledLoopOp.getLoc();
    Value constOne = rewriter.create<ConstantIndexOp>(loc, 1);
    std::array<Value, numWorkgroupDim> workgroups = {constOne, constOne,
                                                     constOne};
    for (auto nw : llvm::enumerate(numWorkgroups)) {
      if (nw.index() >= numWorkgroupDim) break;
      workgroups[nw.index()] =
          rewriter.create<ConstantIndexOp>(loc, nw.value());
    }
    //  Wrap the linalg.tiled_loops into a gpu Launch op.
    auto launchOp = rewriter.create<gpu::LaunchOp>(
        loc, workgroups[0], workgroups[1], workgroups[2], constOne, constOne,
        constOne);
    rewriter.updateRootInPlace(tiledLoopOp, [&] {
      tiledLoopOp->moveBefore(&launchOp.body().front(),
                              launchOp.body().front().begin());
    });

    rewriter.setInsertionPointToEnd(&launchOp.body().front());
    rewriter.create<gpu::TerminatorOp>(loc, llvm::None);

    // Register all the buffers used by the linalg.tiled_loop op.
    rewriter.setInsertionPoint(launchOp);
    auto registerBuffer = [&rewriter, loc](Value buffer) {
      if (buffer.getType().isa<MemRefType>()) {
        Value memref = buffer;
        auto elementType = memref.getType().cast<MemRefType>().getElementType();
        auto unrankedType = UnrankedMemRefType::get(elementType, 0);
        Value unrankedMemref =
            rewriter.create<memref::CastOp>(loc, memref, unrankedType);
        rewriter.create<gpu::HostRegisterOp>(loc, unrankedMemref);
      }
    };
    for (Value arg : tiledLoopOp.inputs()) registerBuffer(arg);
    for (Value arg : tiledLoopOp.outputs()) registerBuffer(arg);
    return success();
  }

 private:
  llvm::SmallVector<int64_t, numWorkgroupDim> numWorkgroups;
};

}  // anonymous namespace

namespace mlir {
namespace linalg {

void populateTiledLoopsToGPUPatterns(OwningRewritePatternList& patterns,
                                     ArrayRef<int64_t> numWorkgroups) {
  patterns.add<TiledLoopToGPUPattern>(patterns.getContext(), numWorkgroups);
}

}  // namespace linalg
}  // namespace mlir
