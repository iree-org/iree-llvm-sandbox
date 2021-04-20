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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;  // NOLINT

namespace {

class ConvertToAsyncPass
    : public PassWrapper<ConvertToAsyncPass, FunctionPass> {
 public:
  ConvertToAsyncPass() = default;
  ConvertToAsyncPass(const ConvertToAsyncPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<async::AsyncDialect>();
  }
  void runOnFunction() override;
};

}  // anonymous namespace

void ConvertToAsyncPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  linalg::TiledLoopOp tiledLoopOp;
  // Assume the first level TiledLoop op maps to Async dispatch. In the future
  // there may be some annotations to make it explicit.
  funcOp.walk([&](linalg::TiledLoopOp op) {
    tiledLoopOp = op;
    return mlir::WalkResult::interrupt();
  });
  if (!tiledLoopOp) return;

  assert(tiledLoopOp.getNumResults() == 0 && "expected bufferized TiledLoopOp");

  auto *ctx = funcOp.getContext();
  Location loc = tiledLoopOp.getLoc();
  OpBuilder b(tiledLoopOp);

  // Wrap the linalg.tiled_loops into an async::ExecuteOp.
  // 1. Create the async::GroupType object on which we synchronize.
  Value asyncGroup =
      b.create<async::CreateGroupOp>(loc, async::GroupType::get(ctx));

  // 2. Create an empty executeOp with an empty yield.
  auto noopExec = [&](OpBuilder &executeBuilder, Location executeLoc,
                      ValueRange executeArgs) {};
  auto execute =
      b.create<async::ExecuteOp>(loc, /*resultTypes=*/TypeRange(),
                                 /*dependencies=*/ValueRange(),
                                 /*operands=*/ValueRange(), noopExec);
  OpBuilder::atBlockEnd(execute.getBody())
      .create<async::YieldOp>(loc, ValueRange{});

  // 3. Steal the linalg::TiledLoopOp ops, except the terminator, into the body
  // of the async::ExecuteOp, just before the terminator.
  execute.getBody()->getOperations().splice(
      std::prev(execute.getBody()->end()),
      tiledLoopOp.getBody()->getOperations(),
      tiledLoopOp.getBody()->getOperations().begin(),
      std::prev(tiledLoopOp.getBody()->getOperations().end()));

  // 4. Move the async::ExecuteOp inside the body of the linalg::TiledLoopOp.
  execute->moveBefore(tiledLoopOp.getBody()->getTerminator());

  // 5. Each time linalg::TiledLoopOp spawns a new async::ExecuteOp, the task
  // gets added as a dependency to the group.
  OpBuilder::atBlockTerminator(tiledLoopOp.getBody())
      .create<async::AddToGroupOp>(loc, b.getIndexType(), execute.token(),
                                   asyncGroup);

  // 6. After the linalg::TiledLoopOp, await all async tasks in `asyncGroup`.
  OpBuilder(tiledLoopOp->getNextNode())
      .create<async::AwaitAllOp>(loc, asyncGroup);
}

namespace mlir {
void registerConvertToAsyncPass() {
  PassRegistration<ConvertToAsyncPass> pass(
      "convert-to-async", "Convert function into a asyn CPU kernel.");
}
}  // namespace mlir
