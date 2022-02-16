//===-- for_to_dowhile.cpp - Implement for to do-while tranformation ------*-
// c++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "alp/Transforms/PassDetail.h"
#include "alp/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include <unordered_map>

#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "for-to-dowhile"
using namespace mlir;

namespace {
using scf::ForOp;
using scf::WhileOp;

struct ForLoopLoweringPattern : public OpRewritePattern<ForOp> {
  using OpRewritePattern<ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ForOp forOp,
                                PatternRewriter &rewriter) const override {
    // Generate type signature for the loop-carried values. The induction
    // variable is placed first, followed by the forOp.iterArgs.
    SmallVector<Type, 8> lcvTypes;
    lcvTypes.push_back(forOp.getInductionVar().getType());
    llvm::transform(forOp.getInitArgs(), std::back_inserter(lcvTypes),
                    [&](auto v) { return v.getType(); });

    // Build scf.WhileOp
    SmallVector<Value> initArgs;
    initArgs.push_back(forOp.getLowerBound());
    llvm::append_range(initArgs, forOp.getInitArgs());

    // We need to add an if-else condition to avoid executing the first
    // iteration
    auto shouldWhileExecute = rewriter.create<arith::CmpIOp>(
        forOp.getLoc(), arith::CmpIPredicate::sgt, forOp.getUpperBound(),
        forOp.getLowerBound());

    auto if_while_should_execute = rewriter.create<scf::IfOp>(
        forOp.getLoc(), forOp.getResultTypes(), shouldWhileExecute,
        forOp.getNumIterOperands() > 0);
    rewriter.setInsertionPointToStart(
        &if_while_should_execute.getThenRegion().front());

    // The while-loop should be contained within the then region
    auto whileOp = rewriter.create<WhileOp>(forOp.getLoc(), lcvTypes, initArgs,
                                            forOp->getAttrs());

    llvm::SmallVector<Location, 4> locs(lcvTypes.size(), forOp.getLoc());
    auto *beforeBlock = rewriter.createBlock(
        &whileOp.getBefore(), whileOp.getBefore().begin(), lcvTypes, locs);

    auto *afterBlock = rewriter.createBlock(
        &whileOp.getAfter(), whileOp.getAfter().begin(), lcvTypes, locs);

    // Rewrite uses of the for-loop block arguments to the new while-loop
    // "after" arguments
    for (auto barg : enumerate(forOp.getBody(0)->getArguments()))
      barg.value().replaceAllUsesWith(beforeBlock->getArgument(barg.index()));
    // Inline for-loop body operations into 'before' region (except Yield).
    llvm::SmallVector<Value> nextIterArgs;
    for (auto &arg : llvm::make_early_inc_range(*forOp.getBody())) {
      if (auto yieldOp = dyn_cast<scf::YieldOp>(&arg)) {
        nextIterArgs = yieldOp.getOperands();
      } else {
        arg.moveBefore(beforeBlock, beforeBlock->end());
      }
    }

    // 'before' region contains the loop condition and forwarding of iteration
    // arguments to the 'after' region.
    rewriter.setInsertionPointToEnd(&whileOp.getBefore().front());

    // Add induction variable incrementation
    auto ivIncOp = rewriter.create<arith::AddIOp>(
        whileOp.getLoc(), beforeBlock->getArgument(0), forOp.getStep());
    auto cmpOp = rewriter.create<arith::CmpIOp>(whileOp.getLoc(),
                                                arith::CmpIPredicate::slt,
                                                ivIncOp, forOp.getUpperBound());

    nextIterArgs.insert(nextIterArgs.begin(), ivIncOp.getResult());
    rewriter.create<scf::ConditionOp>(whileOp.getLoc(), cmpOp.getResult(),
                                      nextIterArgs);

    // Inline for-loop body into an executeRegion operation in the "after"
    // region. The return type of the execRegionOp does not contain the
    // iv - yields in the source for-loop contain only iterArgs.

    // SmallVector<Value> yieldOperands;
    rewriter.setInsertionPointToEnd(afterBlock);
    rewriter.create<scf::YieldOp>(whileOp.getLoc(), afterBlock->getArguments());

    llvm::SmallVector<Value, 4> if_values;
    for (auto arg : llvm::enumerate(forOp.getResults())) {
      if_values.push_back(whileOp.getResult(arg.index() + 1));
    }

    if (if_values.size() > 0) {
      rewriter.setInsertionPointAfter(whileOp);
      rewriter.create<scf::YieldOp>(whileOp.getLoc(), if_values);
    }

    if (forOp.getNumIterOperands() > 0) {
      rewriter.setInsertionPointToStart(
          &if_while_should_execute.getElseRegion().front());
      rewriter.create<scf::YieldOp>(whileOp.getLoc(), forOp.getInitArgs());
    }

    rewriter.replaceOp(forOp, if_while_should_execute.getResults());

    return success();
  }
};
struct ForToDoWhileLoop : public ForToDoWhileLoopBase<ForToDoWhileLoop> {
  void runOnOperation() override {
    // Apply on the given function name

    FuncOp funcOp = getOperation();
    if (anchorFuncOpName != funcOp.getName()) {
      return;
    }
    LLVM_DEBUG(llvm::dbgs() << "loop conversion starts\n ");

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<ForLoopLoweringPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<FuncOp>>
mlir::createForToDoWhileLoopPass() {
  return std::make_unique<ForToDoWhileLoop>();
}