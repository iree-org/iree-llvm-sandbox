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
#include "mlir/Transforms/DialectConversion.h"
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

struct GetNumProgramsOpConversion
    : public OpRewritePattern<triton::GetNumProgramsOp> {
  GetNumProgramsOpConversion(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(triton::GetNumProgramsOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    Block::BlockArgListType funcArgs =
        funcOp.getFunctionBody().front().getArguments();
    auto dim = static_cast<uint32_t>(op.getAxis());
    Value programIdArg = funcArgs[dim + 3];
    rewriter.replaceOp(op, programIdArg);
    return success();
  }
};

struct GetProgramIdOpConversion
    : public OpRewritePattern<triton::GetProgramIdOp> {
  GetProgramIdOpConversion(MLIRContext *context, PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(triton::GetProgramIdOp op,
                                PatternRewriter &rewriter) const override {
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    Block::BlockArgListType funcArgs =
        funcOp.getFunctionBody().front().getArguments();
    auto dim = static_cast<uint32_t>(op.getAxis());
    Value programIdArg = funcArgs[dim];
    rewriter.replaceOp(op, programIdArg);
    return success();
  }
};

void addArgumentsToFront(FunctionOpInterface op, RewriterBase &rewriter,
                         TypeRange additionalArgs) {
  MLIRContext *context = rewriter.getContext();

  // Compute new function type: add new arguments at the beginning.
  TypeRange oldArgTypes = op.getArgumentTypes();
  SmallVector<Type> newArgTypes(additionalArgs.begin(), additionalArgs.end());
  newArgTypes.append(oldArgTypes.begin(), oldArgTypes.end());
  TypeRange resultTypes = op.getResultTypes();
  auto newType =
      FunctionType::get(rewriter.getContext(), newArgTypes, resultTypes);

  // Compute new argument attributes: add two empty dicts for the new args.
  SmallVector<DictionaryAttr> newArgAttrs(additionalArgs.size(),
                                          DictionaryAttr::get(context));
  op.getAllArgAttrs(newArgAttrs);

  // Update function type and argument attributes in-place.
  rewriter.updateRootInPlace(op, [&] {
    op.setType(newType);
    op.setAllArgAttrs(newArgAttrs);
  });

  // Compute new arguments for the entry block of the body.
  Block *oldBlock = &op.getFunctionBody().front();
  SmallVector<Location> locs(additionalArgs.size(), op.getLoc());
  for (BlockArgument arg : oldBlock->getArguments()) {
    locs.push_back(arg.getLoc());
  }

  // Create a new block with the new signature and merge the old one into it.
  Block *newBlock = rewriter.createBlock(oldBlock, newArgTypes, locs);
  rewriter.replaceAllUsesWith(oldBlock, newBlock);
  ArrayRef<BlockArgument> forwardArgs =
      newBlock->getArguments().drop_front(additionalArgs.size());
  rewriter.mergeBlocks(oldBlock, newBlock, forwardArgs);
}

void addGridArguments(FunctionOpInterface op, RewriterBase &rewriter) {
  // New argument types: add three dimensions x (program_id, num_programs).
  Type i32 = rewriter.getI32Type();
  SmallVector<Type, 6> gridArgs = {i32, i32, i32,  // program_id per dimension
                                   i32, i32, i32}; // num_programs per dimension

  addArgumentsToFront(op, rewriter, gridArgs);
}
} // namespace

void mlir::populateTritonSPMDToFuncArgsConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<
      // clang-format off
      GetNumProgramsOpConversion,
      GetProgramIdOpConversion
      // clang-format on
      >(patterns.getContext());
}

void ConvertTritonSPMDToFuncArgsPass::runOnOperation() {
  auto module = getOperation();
  MLIRContext *context = &getContext();

  // Add grid arguments to all functions.
  IRRewriter rewriter(context);
  for (auto op : module.getOps<FunctionOpInterface>()) {
    if (!op.isExternal())
      addGridArguments(op, rewriter);
  }

  // Add grid arguments to all call ops. This must happen after adding the
  // arguments to the functions since it uses the newly added arguments as
  // arguments of the call ops.
  module->walk([](CallOpInterface op) {
    auto funcOp = op->getParentOfType<FunctionOpInterface>();
    llvm::ArrayRef<BlockArgument> gridArgs =
        funcOp.getArguments().take_front(6);
    SmallVector<Value, 12> newOperands(gridArgs.begin(), gridArgs.end());
    ValueRange originalArgs = op.getArgOperands();
    newOperands.append(originalArgs.begin(), originalArgs.end());
    op->setOperands(newOperands);
  });

  // Convert the SPMD ops in the Triton dialect to accesses to the corresponding
  // function arguments.
  RewritePatternSet patterns(&getContext());
  populateTritonSPMDToFuncArgsConversionPatterns(patterns);

  // Mark the SPMD ops as illegal; everything else is legal.
  ConversionTarget target(getContext());
  target.addLegalDialect<TritonDialect>();
  target.addIllegalOp<triton::GetNumProgramsOp, triton::GetProgramIdOp>();
  target.addLegalOp<ModuleOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertTritonSPMDToFuncArgsPass() {
  return std::make_unique<ConvertTritonSPMDToFuncArgsPass>();
}
