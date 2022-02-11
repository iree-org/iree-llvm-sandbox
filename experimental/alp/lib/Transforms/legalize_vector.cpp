//===-- legalize_vector.cpp - Legalize Vectors Pass ------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "alp/Transforms/PassDetail.h"
#include "alp/Transforms/Passes.h"

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/IR/BlockAndValueMapping.h"
//#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include <unordered_map>

#define DEBUG_TYPE "legalize-vector"

using namespace mlir;
using namespace mlir::vector;

struct ForOpVectorProgates : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const final {

    InsertStridedSliceOp insertOp =
        forOp.getIterOperands()[0].getDefiningOp<InsertStridedSliceOp>();

    if (!insertOp) {
      return failure();
    }

    SmallVector<Value> newIterOperands;

    newIterOperands.push_back(insertOp.dest());
    auto loc = forOp.getLoc();

    for (auto op : forOp.getIterOperands()) {
      if (op == insertOp) {
        continue;
      }
      newIterOperands.push_back(op);
    }
    newIterOperands.push_back(insertOp.source());
    int64_t working_offset1 = extractFromI64ArrayAttr(insertOp.offsets())[0];
    int64_t working_offset2 = extractFromI64ArrayAttr(insertOp.offsets())[1];

    Block &oldBlock = forOp.getRegion().front();
    oldBlock.addArgument(insertOp.source().getType(), loc);

    scf::ForOp newForOp = rewriter.create<scf::ForOp>(
        loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
        newIterOperands);

    Block &newBlock = newForOp.getRegion().front();
    SmallVector<Value, 4> extraYieldOperands;

    oldBlock.walk([&](Operation *instr) {
      if (auto extractOp = dyn_cast<ExtractStridedSliceOp>(instr)) {
        if (std::find(oldBlock.getArguments().begin(),
                      oldBlock.getArguments().end(),
                      extractOp.vector()) != oldBlock.getArguments().end()) {
          // if offsets are the same
          int64_t offset1 = extractFromI64ArrayAttr(extractOp.offsets())[0];
          int64_t offset2 = extractFromI64ArrayAttr(extractOp.offsets())[1];
          if (offset1 == working_offset1 && offset2 == working_offset2) {
            extractOp.getResult().replaceAllUsesWith(
                oldBlock.getArguments().back());
            rewriter.eraseOp(extractOp);
          }
        }
      } else if (auto insertOp = dyn_cast<InsertStridedSliceOp>(instr)) {
        // if (std::find(forOp.getResults().begin(), forOp.getResults().end(),
        // insertOp.dest()) != forOp.getResults().end()){
        int64_t offset1 = extractFromI64ArrayAttr(insertOp.offsets())[0];
        int64_t offset2 = extractFromI64ArrayAttr(insertOp.offsets())[1];
        if (offset1 == working_offset1 && offset2 == working_offset2) {
          insertOp.getResult().replaceAllUsesWith(insertOp.dest());
          extraYieldOperands.push_back(insertOp.source());
          rewriter.eraseOp(insertOp);
        }
        // }
      }
    });

    SmallVector<Value, 4> newBlockTransferArgs(newBlock.getArguments().begin(),
                                               newBlock.getArguments().end());
    rewriter.mergeBlocks(&oldBlock, &newBlock, newBlockTransferArgs);
    auto clonedYieldOp = cast<scf::YieldOp>(newBlock.getTerminator());
    SmallVector<Value> newYieldOperands = clonedYieldOp.getOperands();
    for (auto val : extraYieldOperands) {
      newYieldOperands.push_back(val);
    }
    rewriter.setInsertionPoint(clonedYieldOp);
    rewriter.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
    rewriter.eraseOp(clonedYieldOp);

    forOp.getResult(0).replaceAllUsesWith(newForOp.getResult(0));

    SmallVector<Value, 4> newResults;
    for (unsigned i = 0; i < forOp.getNumResults(); i++) {
      newResults.push_back(newForOp.getResult(i));
    }
    rewriter.replaceOp(forOp, newResults);
    // Get the users of the original result
    auto users = newForOp.getResult(0).getUsers();

    // And now let's change this
    for (Operation *user : users) {
      if (auto extractOp = dyn_cast<ExtractStridedSliceOp>(user)) {
        int64_t offset1 = extractFromI64ArrayAttr(extractOp.offsets())[0];
        int64_t offset2 = extractFromI64ArrayAttr(extractOp.offsets())[1];
        if (offset1 == working_offset1 && offset2 == working_offset2) {
          extractOp.getResult().replaceAllUsesWith(
              newForOp.getResults().back());
          rewriter.eraseOp(extractOp);
        }
      }
    }
    return success();
  }
};

struct LegalizePass : public LegalizePassBase<LegalizePass> {
  LegalizePass() = default;
  LegalizePass(const LegalizePass &pass) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    if (getOperation().getName() != "kernel") {
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "legalize starts\n ");

    RewritePatternSet patterns(ctx);
    populateVectorUnrollPatterns(
        patterns,
        UnrollVectorOptions().setNativeShapeFn(getShape).setFilterConstraint(
            filter));
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));

    // Now there will be a lot of:
    // - transfer_read/insert_strided_slices
    // - transfer_write/extract_strided_slices
    // Next step is to remove as many {insert,extract}_strided_slices as we can,
    // especially across loop blocks
    RewritePatternSet extra_patterns(ctx);
    extra_patterns.add<ForOpVectorProgates>(ctx);
    (void)applyPatternsAndFoldGreedily(getOperation(),
                                       std::move(extra_patterns));
  }

private:
  // Return the target shape based on op type.
  static Optional<SmallVector<int64_t, 4>> getShape(Operation *op) {
    if (isa<arith::MulFOp, arith::AddFOp, SelectOp, arith::CmpFOp>(op))
      return SmallVector<int64_t, 4>(2, 4);
    if (isa<vector::ContractionOp>(op)) {
      return SmallVector<int64_t, 4>{4, 4, 1};
    }
    // For transfer ops, just propagate the shape coming from
    // InsertStridedSlices/ExtractStridedSlices.
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      VectorType dstVec;
      for (Operation *users : readOp->getUsers()) {
        auto extract = dyn_cast<ExtractStridedSliceOp>(users);
        scf::ForOp loop = dyn_cast<scf::ForOp>(users);
        if (loop) {
          if (loop.getIterOperands()[0] == readOp) {
            OpOperand &val_arg = loop.getIterOpOperands()[0];
            auto val = loop.getRegionIterArgForOpOperand(val_arg);
            for (Operation *arg_users : val.getUsers()) {
              extract = dyn_cast<ExtractStridedSliceOp>(arg_users);
            }
          }
        }
        if (!extract) {
          return llvm::None;
        }
        auto vecType = extract.getResult().getType().cast<VectorType>();
        if (dstVec && dstVec != vecType)
          return llvm::None;
        dstVec = vecType;
      }
      return SmallVector<int64_t, 4>(dstVec.getShape().begin(),
                                     dstVec.getShape().end());
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      auto insert = writeOp.vector().getDefiningOp<InsertStridedSliceOp>();
      auto loop = writeOp.vector().getDefiningOp<scf::ForOp>();
      if (loop) {
        auto yieldOp = cast<scf::YieldOp>(&loop.getRegion().front().back());
        insert = yieldOp.getOperand(0).getDefiningOp<InsertStridedSliceOp>();
      }
      if (!insert)
        return llvm::None;
      ArrayRef<int64_t> shape = insert.getSourceVectorType().getShape();
      return SmallVector<int64_t, 4>(shape.begin(), shape.end());
    }
    return llvm::None;
  }

  static LogicalResult filter(Operation *op) {
    return success(isa<arith::AddFOp, SelectOp, arith::CmpFOp, ContractionOp,
                       TransferReadOp, TransferWriteOp>(op));
  }
};

std::unique_ptr<mlir::OperationPass<FuncOp>> mlir::createLegalizePass() {
  return std::make_unique<LegalizePass>();
}
