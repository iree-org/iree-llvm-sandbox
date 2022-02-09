//===-- PDL.cpp - Interoperability with PDL -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PDL.h"

#include "Transforms/Functional.h"
#include "mlir/Dialect/PDL/IR/PDLOps.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Rewrite/PatternApplicator.h"

namespace mlir {
namespace linalg {

/// Return ops that match any of the patterns.
static SmallVector<LinalgOp> getMatchingOps(
    Operation *parent, const FrozenRewritePatternSet &patterns) {
  PatternApplicator applicator(patterns);
  applicator.applyDefaultCostModel();

  // TODO: The C++ functional API needs better interoperability with PDL.
  return functional::applyForEachIn(
      parent,
      [&](Operation *op, PatternRewriter &rewriter) -> FailureOr<LinalgOp> {
        if (succeeded(applicator.matchAndRewrite(op, rewriter)))
          if (auto linalgOp = dyn_cast<LinalgOp>(op)) return linalgOp;
        return failure();
      });
}

/// Hook for PDL driver to check if an operation (`value`) is directly nested in
/// a function with the name provided as constant parameter.
/// TODO: PDL needs user-defined "questions".
static LogicalResult nestedInFunc(PDLValue value, ArrayAttr constantParams,
                                  PatternRewriter &rewriter) {
  auto *operation = value.cast<Operation *>();
  auto func = operation->getParentOfType<FuncOp>();
  assert(constantParams.size() == 1 &&
         "expected a constant param with function name");
  auto functionSymbol = constantParams[0].dyn_cast<SymbolRefAttr>();
  assert(functionSymbol && "expected a function name");

  if (!func)
    return rewriter.notifyMatchFailure(operation, "not nested in a function");
  return success(functionSymbol.getLeafReference() == func.getName());
}

/// PDL rewrite hook that does nothing.
static void noOpRewriter(ArrayRef<PDLValue> args, ArrayAttr constantParams,
                         PatternRewriter &rewriter, PDLResultList &results) {
  assert(args.size() == 1 && "expected one argument");
#ifndef NDEBUG
  args.front().cast<Operation *>()->setAttr("linalg_transform.matched",
                                            rewriter.getUnitAttr());
#endif
}

FailureOr<SmallVector<LinalgOp>> findMatchingOps(Operation *op,
                                                 SymbolRefAttr pattern,
                                                 ModuleOp module) {
  auto patternOp = module.lookupSymbol<pdl::PatternOp>(pattern);
  if (!patternOp)
    return {op->emitError("could not find a pattern named: ") << pattern};

  // Clone the pattern operation into the temporary module used by the driver
  // as it might be referenced multiple times.
  OwningOpRef<ModuleOp> pdlModuleOp = ModuleOp::create(patternOp.getLoc());
  OpBuilder::atBlockBegin(&pdlModuleOp->body().front()).clone(*patternOp);

  // Build the PDL module.
  PDLPatternModule pdlModule(std::move(pdlModuleOp));
  pdlModule.registerConstraintFunction("nestedInFunc", nestedInFunc);
  pdlModule.registerRewriteFunction("linalg_transform.apply", noOpRewriter);

  RewritePatternSet patterns(std::move(pdlModule));
  return getMatchingOps(module, std::move(patterns));
}

}  // namespace linalg
}  // namespace mlir
