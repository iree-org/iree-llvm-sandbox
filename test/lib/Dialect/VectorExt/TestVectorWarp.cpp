//===- TestVectorWarp.cpp - Test vector warp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/VectorExt/VectorExtOps.h"
#include "Dialect/VectorExt/VectorExtWarpUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::vector;
using namespace mlir::vector_ext;

static const int64_t kSharedMemorySpace = 3;

/// Allocate shared memory for a single warp to test lowering of
/// WarpExecuteOnLane0Op.
static Value allocateGlobalSharedMemory(Location loc, OpBuilder &builder,
                                        WarpExecuteOnLane0Op warpOp,
                                        Type type) {
  // Compute type of shared memory buffer.
  MemRefType memrefType;
  if (auto vectorType = type.dyn_cast<VectorType>()) {
    memrefType =
        MemRefType::get(vectorType.getShape(), vectorType.getElementType(), {},
                        kSharedMemorySpace);
  } else {
    memrefType = MemRefType::get({1}, type, {}, kSharedMemorySpace);
  }

  // Get symbol table holding all shared memory globals.
  ModuleOp moduleOp = warpOp->getParentOfType<ModuleOp>();
  SymbolTable symbolTable(moduleOp);

  // Create a pretty name.
  SmallString<64> buf;
  llvm::raw_svector_ostream os(buf);
  interleave(memrefType.getShape(), os, "x");
  os << "x" << memrefType.getElementType();
  std::string symbolName = (Twine("__shared_") + os.str()).str();

  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPoint(moduleOp);
  auto global = builder.create<memref::GlobalOp>(
      loc,
      /*sym_name=*/symbolName,
      /*sym_visibility=*/builder.getStringAttr("private"),
      /*type=*/memrefType,
      /*initial_value=*/Attribute(),
      /*constant=*/false,
      /*alignment=*/IntegerAttr());
  symbolTable.insert(global);
  // The symbol table inserts at the end of the module, but globals are a bit
  // nicer if they are at the beginning.
  global->moveBefore(&moduleOp.front());

  builder.restoreInsertionPoint(ip);
  return builder.create<memref::GetGlobalOp>(loc, memrefType, symbolName);
}

namespace {

struct TestVectorWarp
    : public PassWrapper<TestVectorWarp, OperationPass<FuncOp>> {

  TestVectorWarp() = default;
  TestVectorWarp(const TestVectorWarp &pass) {}

  StringRef getArgument() const final { return "test-vector-warp-distribute"; }
  StringRef getDescription() const final {
    return "Test vector warp transformations";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<VectorDialect, VectorExtDialect>();
  }

  Option<bool> distributeTransferWriteOps{
      *this, "distribute-transfer-write",
      llvm::cl::desc("Test distribution of transfer write"),
      llvm::cl::init(false)};

  Option<bool> hoistUniform{*this, "hoist-uniform",
                            llvm::cl::desc("Test hoist uniform"),
                            llvm::cl::init(false)};

  Option<bool> propagateDistribution{
      *this, "propagate-distribution",
      llvm::cl::desc("Test distribution propgation"), llvm::cl::init(false)};

  Option<bool> rewriteWarpOpsToScfIf{
      *this, "rewrite-warp-ops-to-scf-if",
      llvm::cl::desc("Test rewriting of warp_execute_on_lane_0 to scf.if")};

  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    funcOp.walk([&](Operation *op) {
      if (auto warpOp = dyn_cast<WarpExecuteOnLane0Op>(op)) {
        if (hoistUniform) {
          moveScalarUniformCode(warpOp);
        }
        WalkResult::interrupt();
      }
    });
    MLIRContext *ctx = &getContext();
    if (distributeTransferWriteOps) {
      auto distributionFn = [](vector::TransferWriteOp writeOp) {
        // Create a map (d0, d1) -> (d1) to distribute along the inner
        // dimension. Once we support n-d distribution we can add more
        // complex cases.
        int64_t vecRank = writeOp.getVectorType().getRank();
        OpBuilder builder(writeOp.getContext());
        auto map =
            AffineMap::get(vecRank, 0, builder.getAffineDimExpr(vecRank - 1));
        return map;
      };
      RewritePatternSet patterns(ctx);
      populateDistributeTransferWriteOpPatterns(patterns, distributionFn);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
    if (propagateDistribution) {
      RewritePatternSet patterns(ctx);
      vector_ext::populatePropagateVectorDistributionPatterns(patterns);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
    if (rewriteWarpOpsToScfIf) {
      RewritePatternSet patterns(ctx);
      vector_ext::populateWarpExecuteOnLane0OpToScfForPattern(
          patterns, allocateGlobalSharedMemory);
      (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
    }
  }
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestVectorWarps() { PassRegistration<TestVectorWarp>(); }
} // namespace test_ext
} // namespace mlir
