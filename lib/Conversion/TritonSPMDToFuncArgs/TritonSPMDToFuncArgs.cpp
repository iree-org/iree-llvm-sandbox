//===-- TritonSPMDToFuncArgs.cpp - Triton SPMD ops to func args -*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TritonSPMDToFuncArgs/TritonSPMDToFuncArgs.h"

#include "../PassDetail.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h"

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

/// Trait for finding the call and return op types for a given func op type.
template <class FuncOpType>
struct FuncTypeTraits;

template <>
struct FuncTypeTraits<func::FuncOp> {
  using ReturnOp = func::ReturnOp;
  using CallOp = func::CallOp;
};

template <>
struct FuncTypeTraits<triton::FuncOp> {
  using ReturnOp = triton::ReturnOp;
  using CallOp = triton::CallOp;
};

template <class FuncOpType>
void buildGridFunctionBody(FuncOpType gridFunc, FuncOpType kernelFunc,
                           RewriterBase &rewriter) {
  assert(gridFunc.getResultTypes().empty() &&
         "kernels can't have a return value so neither can their grids");
  assert(kernelFunc.getResultTypes().empty() &&
         "kernels can't have a return value");

  Location loc = gridFunc->getLoc();
  Type i32 = rewriter.getI32Type();
  Type idx = rewriter.getIndexType();
  using ReturnOp = typename FuncTypeTraits<FuncOpType>::ReturnOp;
  using CallOp = typename FuncTypeTraits<FuncOpType>::CallOp;

  Block *entryBlock = gridFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  // New argument types: three dimensions x num_programs.
  SmallVector<Type, 3> gridArgs = {i32, i32, i32};
  addArgumentsToFront(gridFunc, rewriter, gridArgs);

  Value xub = gridFunc.getArgument(0);
  Value yub = gridFunc.getArgument(1);
  Value zub = gridFunc.getArgument(2);

  // Prepare bounds and steps for scf::ParallelOp.
  Value xubIdx = rewriter.create<arith::IndexCastOp>(loc, idx, xub);
  Value yubIdx = rewriter.create<arith::IndexCastOp>(loc, idx, yub);
  Value zubIdx = rewriter.create<arith::IndexCastOp>(loc, idx, zub);

  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);

  // Add scf::ParallelOp that iterates over the grid.
  rewriter.create<scf::ParallelOp>(
      loc, /*lowerBounds=*/ValueRange{zero, zero, zero},
      /*upperBounds=*/ValueRange{xubIdx, yubIdx, zubIdx},
      /*steps=*/ValueRange{one, one, one},
      /*bodyBuilderFn=*/
      [&](OpBuilder &builder, Location loc, ValueRange idx) {
        // Cast back to i32.
        Value x = builder.create<arith::IndexCastOp>(loc, i32, idx[0]);
        Value y = builder.create<arith::IndexCastOp>(loc, i32, idx[1]);
        Value z = builder.create<arith::IndexCastOp>(loc, i32, idx[2]);

        // Prepare call arguments and call the kernel function.
        SmallVector<Value, 12> callArgs = {x, y, z};
        ValueRange forwardArgs = gridFunc.getArguments();
        callArgs.append(forwardArgs.begin(), forwardArgs.end());

        builder.create<CallOp>(loc, kernelFunc, callArgs);

        builder.create<scf::YieldOp>(loc);
      });

  rewriter.create<ReturnOp>(loc);
}

template <class FuncOpType>
void buildGridFunctionBody(FuncOpType gridFunc, FunctionOpInterface kernelFunc,
                           RewriterBase &rewriter) {
  assert(llvm::isa<FuncOpType>(kernelFunc) &&
         "found kernel func op that doesn't match its grid func op");
  buildGridFunctionBody(gridFunc, llvm::cast<FuncOpType>(kernelFunc), rewriter);
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

  // Add grid arguments to all functions and remember kernel functions and grid
  // function stubs.
  SmallVector<FunctionOpInterface> gridFuncs;
  SmallVector<FunctionOpInterface> kernelFuncs;
  IRRewriter rewriter(context);
  for (auto op : module.getOps<FunctionOpInterface>()) {
    if (!op.isExternal()) {
      // If the function doesn't return anything, it may be a kernel function,
      // so we add a grid wrapper function.
      if (op.getResultTypes().empty()) {
        // Clone the function here under a new name and remember it for later.
        auto opClone =
            llvm::cast<FunctionOpInterface>(op->cloneWithoutRegions());
        opClone.setName((opClone.getName() + Twine("_grid")).str());
        gridFuncs.push_back(opClone);
        kernelFuncs.push_back(op);
      }

      // Add grid arguments to all functions.
      addGridArguments(op, rewriter);
    }
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

  // Add the grid function we cloned above to the current module and populate
  // their bodies.
  for (auto [gridFunc, kernelFunc] : llvm::zip(gridFuncs, kernelFuncs)) {
    module.push_back(gridFunc);
    llvm::TypeSwitch<Operation *>(gridFunc)
        .Case<func::FuncOp, triton::FuncOp>(
            [&, kernelFunc = kernelFunc](auto gridFunc) {
              buildGridFunctionBody(gridFunc, kernelFunc, rewriter);
            })
        .Default(
            [](auto) { assert(false && "encountered unexpected operation"); });
  }

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
