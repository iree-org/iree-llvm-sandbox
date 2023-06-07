//===-- TritonSPMDToFuncArgs.cpp - Triton SPMD ops to func args -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TritonSPMDToFuncArgs/TritonSPMDToFuncArgs.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::func;
using namespace triton;

namespace {
struct ConvertTritonSPMDToFuncArgsPass
    : public ConvertTritonSPMDToFuncArgsBase<ConvertTritonSPMDToFuncArgsPass> {
  void runOnOperation() override;
};

struct TrivialPatternRewriter : public PatternRewriter {
public:
  explicit TrivialPatternRewriter(MLIRContext *context)
      : PatternRewriter(context) {}
};

void addGridArguments(FunctionOpInterface op, PatternRewriter &rewriter) {
  MLIRContext *context = rewriter.getContext();

  // New argument types: add two arguments for program_id and num_programs.
  Type i32 = rewriter.getI32Type();
  ArrayRef<Type> gridArgs = {i32, i32, i32,  // program_id per dimension
                             i32, i32, i32}; // num_programs per dimension

  // Compute new function type: add new arguments at the beginning.
  TypeRange oldArgTypes = op.getArgumentTypes();
  SmallVector<Type> newArgTypes(gridArgs.begin(), gridArgs.end());
  newArgTypes.append(oldArgTypes.begin(), oldArgTypes.end());
  TypeRange resultTypes = op->getResultTypes();
  auto newType =
      FunctionType::get(rewriter.getContext(), newArgTypes, resultTypes);

  // Compute new argument attributes: add two empty dicts for the new args.
  ArrayRef<Attribute> oldArgAttrs = op.getArgAttrsAttr().getValue();
  SmallVector<Attribute> newArgAttrs(gridArgs.size(),
                                     DictionaryAttr::get(context));
  newArgAttrs.append(oldArgAttrs.begin(), oldArgAttrs.end());
  auto newArgAttrsAttr = ArrayAttr::get(context, newArgAttrs);

  // Update function type and argument attributes in-place.
  op.setType(newType);
  op.setArgAttrsAttr(newArgAttrsAttr);

  // Compute new arguments for the entry block of the body.
  Block *oldBlock = &op.getFunctionBody().front();
  SmallVector<Location> locs(gridArgs.size(), op.getLoc());
  for (BlockArgument arg : oldBlock->getArguments()) {
    locs.push_back(arg.getLoc());
  }

  // Create a new block with the new signature and merge the old one into it.
  Block *newBlock = rewriter.createBlock(oldBlock, newArgTypes, locs);
  oldBlock->replaceAllUsesWith(newBlock);
  ArrayRef<BlockArgument> forwardArgs =
      newBlock->getArguments().drop_front(gridArgs.size());
  rewriter.mergeBlocks(oldBlock, newBlock, forwardArgs);
}
} // namespace

void ConvertTritonSPMDToFuncArgsPass::runOnOperation() {
  auto module = getOperation();
  MLIRContext *context = &getContext();

  // Add grid arguments to all functions.
  TrivialPatternRewriter rewriter(context);
  for (auto &op : module.getBodyRegion().front()) {
    if (auto funcOp = llvm::dyn_cast<FunctionOpInterface>(&op)) {
      if (!funcOp.isExternal())
        addGridArguments(funcOp, rewriter);
    }
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTritonSPMDToFuncArgsPass() {
  return std::make_unique<ConvertTritonSPMDToFuncArgsPass>();
}
