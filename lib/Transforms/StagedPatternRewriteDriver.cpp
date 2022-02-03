//===- StagedPatternRewriteDriver.cpp - A staged rewriter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements mlir::applyStagedPatternsAndCanonicalize.
//
//===----------------------------------------------------------------------===//

#include "Transforms/StagedPatternRewriteDriver.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ScopedPrinter.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

#define DEBUG_TYPE "staged-rewrite"

namespace {

#ifndef NDEBUG
const char *logLineComment =
    "//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++//\n";
#endif

struct WorkList {
  void addToWorklist(Operation *op) {
    // Check to see if the worklist already contains this op.
    if (worklistMap.count(op))
      return;

    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  Operation *popFromWorklist() {
    auto *op = worklist.back();
    worklist.pop_back();
    // This operation is no longer in the worklist, keep worklistMap up to date.
    if (op)
      worklistMap.erase(op);
    return op;
  }

  void removeFromWorklist(Operation *op) {
    auto it = worklistMap.find(op);
    if (it != worklistMap.end()) {
      assert(worklist[it->second] == op && "malformed worklist data structure");
      worklist[it->second] = nullptr;
      worklistMap.erase(it);
    }
  }

  bool empty() { return worklist.empty(); }

  size_t size() { return worklist.size(); }

  template <typename IterT>
  void assign(IterT begin, IterT end) {
    worklist.assign(begin, end);
    // TODO: reverse if we need the reverse order.
    for (size_t i = 0, e = worklist.size(); i != e; ++i)
      worklistMap[worklist[i]] = i;
  }

  /// List of newly created ops at each round.
  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;
};

class SinglePatternListener : OpBuilder::Listener {
public:
  SinglePatternListener() {}

protected:
  /// Notification handler for when an operation is inserted into the builder.
  /// `op` is the operation that was inserted.
  /// Override OpBuilder::Listener::notifyOperationInserted.
  void notifyOperationInserted(Operation *op) override {
    LLVM_DEBUG({
      logger.startLine() << "** OperationInserted  : '" << op->getName() << "'("
                         << op << ")\n";
    });
    worklist.addToWorklist(op);
  }

  /// Notification handler for when a block is created using the builder.
  /// `block` is the block that was created.
  /// Override OpBuilder::Listener::notifyBlockCreated.
  virtual void notifyBlockCreated(Block *block) override {
    LLVM_DEBUG({
      logger.startLine() << "** BlockCreated  : '"
                         << block->getParentOp()->getName() << "'("
                         << *block->getParentOp() << ")\n";
    });
  }

  WorkList worklist;

public: // temporary to easily add debug messages from callers.
#ifndef NDEBUG
        /// A logger used to emit information during the application process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};

/// This is a simple driver for the PatternMatcher to apply patterns and perform
/// folding on a single op. It repeatedly applies locally optimal patterns.
class SinglePatternRewriteDriver : public PatternRewriter,
                                   public SinglePatternListener {
public:
  explicit SinglePatternRewriteDriver(ArrayRef<Operation *> roots,
                                      const FrozenRewritePatternSet &patterns)
      : PatternRewriter(roots.front()->getContext()), matcher(patterns) {
    // Apply a simple cost model based solely on pattern benefit.
    matcher.applyDefaultCostModel();
    worklist.assign(roots.begin(), roots.end());
  }

  /// During one local round, we traverse the list of ops that currently need
  /// processing and we apply at most one pattern that has not yet been applied.
  /// The algorithm proceeds as follows:
  ///   1. if no patterns apply, an op is just removed from the worklist;
  ///   2. if no patterns apply to any op, the entire application is considered
  ///   a failure;
  ///   3. atm, one pattern can only ever apply to one op.
  // TODO: Point 3. may be too constraining as we likely do want application of
  // a pattern to independent ops that match (otherwise, we would need to insert
  // as many pattern clones as there are ops that match it, doing the matching
  // upfront).
  // In the future, we may want to target operations more specifically in
  // transformations, "the first op that matches" being only one of the
  // possibilities.
  LogicalResult doOneLocalRound() {
    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << logLineComment;
      logger.startLine() << "Start staged pattern round : " << roundNum
                         << "\n\n";
    });
    ++roundNum;

    if (worklist.empty())
      return failure();

    // Reset the ops to process to prepare for next round.
    LogicalResult res = failure();

    // We want to iterate over all the ops to process this round, apply one
    // pattern then exit.
    do {
      Operation *op = worklist.popFromWorklist();

      LLVM_DEBUG({
        logger.getOStream() << "\n";
        logger.startLine() << logLineComment;
        logger.startLine() << "Staged-processing operation : '" << op->getName()
                           << "'(" << op << ") {\n";
        logger.indent();

        // If the operation has no regions, just print it here.
        if (op->getNumRegions() == 0) {
          op->print(
              logger.startLine(),
              OpPrintingFlags().printGenericOpForm().elideLargeElementsAttrs());
          logger.getOStream() << "\n\n";
        }
      });

      // Only apply one pattern of the list.
      bool appliedThisRound = false;
      auto canApply = [&](const Pattern &pattern) {
        bool res = !appliedPatterns.contains(&pattern);
        LLVM_DEBUG({
          logger.startLine()
              << "** canApply  : '" << &pattern << " -> " << res << ")\n";
        });
        return res;
      };

      auto onFailure = [&](const Pattern &pattern) {
        LLVM_DEBUG({
          logger.startLine() << "** failed to apply  : '" << &pattern << ")\n";
        });
      };

      auto onSuccess = [&](const Pattern &pattern) {
        appliedThisRound = true;
        appliedPatterns.insert(&pattern);
        LLVM_DEBUG(
            { logger.startLine() << "** Applied  : '" << &pattern << ")\n"; });
        return success();
      };

      if (succeeded(matcher.matchAndRewrite(op, *this, canApply, onFailure,
                                            onSuccess)))
        res = success();

      if (appliedThisRound)
        break;
    } while (!worklist.empty());

    return res;
  }

protected:
  /// If an operation is about to be removed, mark it so that we can let clients
  /// know.
  void notifyOperationRemoved(Operation *op) override {
    LLVM_DEBUG({
      logger.startLine() << "** OperationRemoved  : '" << op->getName() << "'("
                         << op << ")\n";
    });
  }

  // When a root is going to be replaced, its removal will be notified as well.
  // So there is nothing to do here.
  void notifyRootReplaced(Operation *op) override {
    LLVM_DEBUG({
      logger.startLine() << "** OperationRemoved  : '" << op->getName() << "'("
                         << op << ")\n";
    });
    llvm_unreachable("Unsupported notifyRootReplaced");
  }

  /// This method is used to notify the rewriter that an in-place operation
  /// modification is about to happen. A call to this function *must* be
  /// followed by a call to either `finalizeRootUpdate` or `cancelRootUpdate`.
  /// This is a minor efficiency win (it avoids creating a new operation and
  /// removing the old one) but also often allows simpler code in the client.
  virtual void startRootUpdate(Operation *op) override {
    LLVM_DEBUG({
      logger.startLine() << "** StartRootUpdate  : '" << op->getName() << "'("
                         << op << ")\n";
    });
  }

  /// This method is used to signal the end of a root update on the given
  /// operation. This can only be called on operations that were provided to a
  /// call to `startRootUpdate`.
  virtual void finalizeRootUpdate(Operation *op) override {
    LLVM_DEBUG({
      logger.startLine() << "** FinalizeRootUpdate  : '" << op->getName()
                         << "'(" << op << ")\n";
    });
    worklist.addToWorklist(op);
  }

private:
  /// Keep track of the round number.
  int64_t roundNum = 0;
  /// The low-level pattern applicator.
  PatternApplicator matcher;
  /// Keep track of patterns applied by this rewriter to only apply them
  /// the specified number of times (i.e. once for now).
  llvm::SmallDenseSet<const Pattern *> appliedPatterns;
};

} // namespace

LogicalResult
mlir::applyStagedPatterns(ArrayRef<Operation *> roots,
                          const FrozenRewritePatternSet &stage1Patterns,
                          const FrozenRewritePatternSet &stage2Patterns,
                          function_ref<LogicalResult(FuncOp)> stage3Lambda) {
  if (roots.empty())
    return success();
  FuncOp funcOp = roots.front()->getParentOfType<FuncOp>();

  // TODO: in the presence of stage2Patterns and stage3Lambda, Operation* are
  // subject to change:
  SinglePatternRewriteDriver driver(roots, stage1Patterns);
  while (succeeded(driver.doOneLocalRound())) {
    LLVM_DEBUG({
      driver.logger.getOStream() << "\n";
      driver.logger.startLine() << logLineComment;
      driver.logger.startLine() << "Start applying stage 2 patterns\n";
      driver.logger.indent();
    });
    if (failed(mlir::applyPatternsAndFoldGreedily(funcOp, stage2Patterns)))
      return failure();
    if (stage3Lambda && failed(stage3Lambda(funcOp)))
      return failure();
  }
  return success();
}
