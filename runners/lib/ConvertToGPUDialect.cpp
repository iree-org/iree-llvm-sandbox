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
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;  // NOLINT

namespace {

class ConvertToGPUPass : public PassWrapper<ConvertToGPUPass, FunctionPass> {
 public:
  ConvertToGPUPass() = default;
  ConvertToGPUPass(const ConvertToGPUPass &pass) {}
  ListOption<int64_t> numWorkgroups{
      *this, "num-workgroups", llvm::cl::MiscFlags::CommaSeparated,
      llvm::cl::desc("Specifies the number of workgroups dispatched.")};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }
  void runOnFunction() override;
};

}  // anonymous namespace

void ConvertToGPUPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  linalg::TiledLoopOp tiledLoopOp;
  // Assume the first level TiledLoop op maps to GPU dispatch. In the future
  // there may be some annotations to make it explicit.
  funcOp.walk([&](linalg::TiledLoopOp op) {
    tiledLoopOp = op;
    return mlir::WalkResult::advance();
  });
  if (!tiledLoopOp) return;
  Location loc = tiledLoopOp.getLoc();
  OpBuilder b(tiledLoopOp);
  Value constOne = b.create<ConstantIndexOp>(loc, 1);
  std::array<Value, 3> workgroups = {constOne, constOne, constOne};
  for (auto nw : llvm::enumerate(numWorkgroups)) {
    if (nw.index() >= 3) break;
    workgroups[nw.index()] = b.create<ConstantIndexOp>(loc, nw.value());
  }
  //  Wrap the linalg.tiled_loops into a gpu Launch op.
  auto launchOp =
      b.create<gpu::LaunchOp>(loc, workgroups[0], workgroups[1], workgroups[2],
                              constOne, constOne, constOne);
  tiledLoopOp->moveBefore(&launchOp.body().front(),
                          launchOp.body().front().begin());

  b.setInsertionPointToEnd(&launchOp.body().front());
  b.create<gpu::TerminatorOp>(loc, llvm::None);

  // Register all the buffers used by the linalg.tiled_loop op.
  b.setInsertionPoint(launchOp);
  auto registerBuffer = [&b, loc](Value buffer) {
    if (buffer.getType().isa<MemRefType>()) {
      Value memref = buffer;
      auto elementType = memref.getType().cast<MemRefType>().getElementType();
      auto unrankedType = UnrankedMemRefType::get(elementType, 0);
      Value unrankedMemref =
          b.create<memref::CastOp>(loc, memref, unrankedType);
      b.create<gpu::HostRegisterOp>(loc, unrankedMemref);
    }
  };
  for (Value arg : tiledLoopOp.inputs()) registerBuffer(arg);
  for (Value arg : tiledLoopOp.outputs()) registerBuffer(arg);
}

namespace mlir {
void registerConvertToGPUPass() {
  PassRegistration<ConvertToGPUPass> pass(
      "convert-to-gpu", "Convert function into a kernel ran on GPU");
}
}  // namespace mlir
