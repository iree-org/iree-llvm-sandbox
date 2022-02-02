//===- TestMaskingUtils.cpp - Utilities for vector masking ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for testing Vector masking utilities.
//
//===----------------------------------------------------------------------===//

#include "Transforms/StagedPatternRewriteDriver.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ScopeExit.h"

using namespace mlir;

constexpr auto kTestStartOpName = "start";
constexpr auto kTestStep1OpName = "step_1";
constexpr auto kTestAttrName = "__test_attr__";

struct IncrementPattern : public RewritePattern {
  explicit IncrementPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    rewriter.startRootUpdate(op);
    auto scope_exit =
        llvm::make_scope_exit([&]() { rewriter.finalizeRootUpdate(op); });

    auto attr = op->getAttrOfType<IntegerAttr>(kTestAttrName);
    // Set 0.
    if (!attr) {
      op->setAttr(kTestAttrName, rewriter.getI32IntegerAttr(0));
      return success();
    }
    // Increment by 1.
    op->setAttr(kTestAttrName,
                rewriter.getI32IntegerAttr(attr.getValue().getSExtValue() + 1));
    return success();
  }
};

struct CloneWithNewNamePattern : public RewritePattern {
  explicit CloneWithNewNamePattern(StringRef oldName, StringRef newName,
                                   MLIRContext *context,
                                   PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), oldName(oldName),
        newName(newName) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getName().getStringRef() != oldName)
      return failure();
    // TODO: support regions if/when needed.
    if (op->getNumResults() > 0)
      return failure();
    // TODO: support successors() if/when needed.
    if (op->getNumSuccessors() > 0)
      return failure();

    OperationState state(op->getLoc(), newName);
    state.addAttributes(op->getAttrs());
    state.addOperands(op->getOperands());
    state.addTypes(op->getResultTypes());

    Operation *newOp = rewriter.createOperation(state);
    if (newOp->getNumResults() == 0)
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, newOp->getResults());
    return success();
  }

  StringRef oldName, newName;
};

namespace {

struct TestStagedPatternRewriteDriver
    : public PassWrapper<TestStagedPatternRewriteDriver,
                         OperationPass<FuncOp>> {

  TestStagedPatternRewriteDriver() = default;
  TestStagedPatternRewriteDriver(const TestStagedPatternRewriteDriver &){};

  StringRef getArgument() const final {
    return "test-staged-pattern-rewrite-driver";
  }

  StringRef getDescription() const final {
    return "Test staged pattern rewrite driver";
  }

  void getDependentDialects(DialectRegistry &registry) const override {}

  void runOnOperation() override {
    SmallVector<Operation *> roots;
    getOperation()->walk([&](Operation *start) {
      if (start->getName().getStringRef() == kTestStartOpName)
        roots.push_back(start);
    });

    MLIRContext *ctx = &getContext();
    RewritePatternSet stage1Patterns(ctx), stage2Patterns(ctx);
    auto scope_exit = llvm::make_scope_exit([&]() {
      (void)applyStagedPatterns(roots, std::move(stage1Patterns),
                                std::move(stage2Patterns));
    });

    if (testCase == 1) {
      stage1Patterns.insert<IncrementPattern>(ctx);
      return;
    }

    if (testCase == 2) {
      stage1Patterns
          .insert<IncrementPattern, IncrementPattern, IncrementPattern>(ctx);
      return;
    }

    if (testCase == 3) {
      stage1Patterns.insert<CloneWithNewNamePattern>(kTestStartOpName,
                                                     kTestStep1OpName, ctx);
      stage1Patterns.insert<IncrementPattern, IncrementPattern>(ctx);
      return;
    }
  }

  Option<int> testCase{*this, "test-case",
                       llvm::cl::desc("Which test case to run")};
};

} // namespace

namespace mlir {
namespace test_ext {
void registerTestStagedPatternRewriteDriver() {
  PassRegistration<TestStagedPatternRewriteDriver>();
}
} // namespace test_ext
} // namespace mlir
