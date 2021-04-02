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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/SmallVector.h"
#include "third_party/llvm/llvm-project/mlir/include/mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;  // NOLINT

namespace {

class ConvertToGPUPass : public PassWrapper<ConvertToGPUPass, FunctionPass> {
 public:
  ConvertToGPUPass() = default;
  ConvertToGPUPass(const ConvertToGPUPass &pass) {}
  Option<std::string> functionName{
      *this, "gpu-func-name",
      llvm::cl::desc("Convert the matching function to GPU"),
      llvm::cl::init("none")};
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<gpu::GPUDialect>();
  }
  void runOnFunction() override;
};

}  // anonymous namespace

void ConvertToGPUPass::runOnFunction() {
  FuncOp funcOp = getFunction();
  if (funcOp.getName() != functionName) return;
  Location loc = funcOp.getLoc();
  Operation *firstOp = &(*funcOp.begin()->begin());
  OpBuilder b(firstOp);
  Value constOne = b.create<ConstantIndexOp>(loc, 1);
  auto launchOp = b.create<gpu::LaunchOp>(loc, constOne, constOne, constOne,
                                          constOne, constOne, constOne);
  Operation *terminator = funcOp.front().getTerminator();

  launchOp.body().front().getOperations().splice(
      launchOp.body().front().begin(), funcOp.front().getOperations(),
      firstOp->getIterator(), terminator->getIterator());
  // TODO: Handle more than one basic block.
  assert(funcOp.getRegion().getBlocks().size() == 1 &&
         "Convert to GPU only supports single block functions right now");

  b.setInsertionPointToEnd(&launchOp.body().front());
  b.create<gpu::TerminatorOp>(loc, llvm::None);

  funcOp.walk([&](memref::AllocOp alloc) {
    alloc->moveBefore(launchOp);
    b.setInsertionPoint(launchOp);
    Value memref = alloc.getResult();
    auto elementType = memref.getType().cast<MemRefType>().getElementType();
    auto unrankedType = UnrankedMemRefType::get(elementType, 0);
    Value unrankedMemref = b.create<memref::CastOp>(loc, memref, unrankedType);
    b.create<gpu::HostRegisterOp>(loc, unrankedMemref);
  });
  funcOp.walk([&](memref::DeallocOp dealloc) { dealloc->moveAfter(launchOp); });

  b.setInsertionPoint(launchOp);
  for (Value arg : funcOp.getArguments()) {
    if (arg.getType().isa<MemRefType>()) {
      Value memref = arg;
      auto elementType = memref.getType().cast<MemRefType>().getElementType();
      auto unrankedType = UnrankedMemRefType::get(elementType, 0);
      Value unrankedMemref =
          b.create<memref::CastOp>(loc, memref, unrankedType);
      b.create<gpu::HostRegisterOp>(loc, unrankedMemref);
    }
  }
}

namespace mlir {
void registerConvertToGPUPass() {
  PassRegistration<ConvertToGPUPass> pass(
      "convert-to-gpu", "Convert function into a kernel ran on GPU");
}
}  // namespace mlir
