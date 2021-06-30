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

#include "mlir/Dialect/Async/IR/Async.h"
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

struct TiledLoopToAsyncPattern : public OpRewritePattern<linalg::TiledLoopOp> {
  using OpRewritePattern<linalg::TiledLoopOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TiledLoopOp tiledLoopOp,
                                PatternRewriter &rewriter) const override {
    assert(tiledLoopOp.getNumResults() == 0 &&
           "expected bufferized TiledLoopOp");
    // Only consider the top level TiledLoop op and skip if it already contains
    // an ExecuteOp.
    if (tiledLoopOp->getParentOfType<linalg::TiledLoopOp>() ||
        llvm::any_of(tiledLoopOp.getBody()->getOperations(),
                     [](Operation &op) { return isa<async::ExecuteOp>(&op); }))
      return failure();
    auto *ctx = tiledLoopOp.getContext();
    Location loc = tiledLoopOp.getLoc();

    // Group size of one as a placeholder. What should be the right size?
    Value groupSize = rewriter.create<ConstantIndexOp>(loc, 1);
    // Wrap the linalg.tiled_loops into an async::ExecuteOp.
    // 1. Create the async::GroupType object on which we synchronize.
    Value asyncGroup = rewriter.create<async::CreateGroupOp>(
        loc, async::GroupType::get(ctx), groupSize);

    // 2. Create an empty executeOp with an empty yield.
    auto noopExec = [&](OpBuilder &executeBuilder, Location executeLoc,
                        ValueRange executeArgs) {};
    auto execute =
        rewriter.create<async::ExecuteOp>(loc, /*resultTypes=*/TypeRange(),
                                          /*dependencies=*/ValueRange(),
                                          /*operands=*/ValueRange(), noopExec);
    rewriter.setInsertionPoint(execute.getBody(), execute.getBody()->end());
    rewriter.create<async::YieldOp>(loc, ValueRange{});

    rewriter.updateRootInPlace(tiledLoopOp, [&] {
      // 3. Steal the linalg::TiledLoopOp ops, except the terminator, into the
      // body of the async::ExecuteOp, just before the terminator.
      execute.getBody()->getOperations().splice(
          std::prev(execute.getBody()->end()),
          tiledLoopOp.getBody()->getOperations(),
          tiledLoopOp.getBody()->getOperations().begin(),
          std::prev(tiledLoopOp.getBody()->getOperations().end()));

      // 4. Move the async::ExecuteOp inside the body of the
      // linalg::TiledLoopOp.
      execute->moveBefore(tiledLoopOp.getBody()->getTerminator());

      // 5. Each time linalg::TiledLoopOp spawns a new async::ExecuteOp, the
      // task
      // gets added as a dependency to the group.
      rewriter.setInsertionPoint(tiledLoopOp.getBody()->getTerminator());
      rewriter.create<async::AddToGroupOp>(loc, rewriter.getIndexType(),
                                           execute.token(), asyncGroup);
    });

    // 6. After the linalg::TiledLoopOp, await all async tasks in `asyncGroup`.
    rewriter.setInsertionPointAfter(tiledLoopOp);
    rewriter.create<async::AwaitAllOp>(loc, asyncGroup);
    return success();
  }
};

}  // anonymous namespace

namespace mlir {
namespace linalg {

void populateTiledLoopToAsyncPatterns(OwningRewritePatternList &patterns) {
  patterns.add<TiledLoopToAsyncPattern>(patterns.getContext());
}

}  // namespace linalg
}  // namespace mlir
