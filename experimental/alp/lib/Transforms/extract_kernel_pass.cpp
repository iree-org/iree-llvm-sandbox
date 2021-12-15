//===-- extract_kernel_pass.cpp - Extract Kernel Pass ------*- c++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "alp/Transforms/PassDetail.h"
#include "alp/Transforms/Passes.h"

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "alp/Transforms/Passes.h"

#include <unordered_map>

#define DEBUG_TYPE "extract-kernel"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

using namespace mlir;

void extract_function(StringRef func_name, Block *block, ModuleOp parentModule,
                      RewriterBase &rewriter, Location loc) {
  SmallVector<Type, 1> Input, Output;

  // Create the function (callee site) with an empty block
  rewriter.setInsertionPointToStart(parentModule.getBody());
  auto func_op = rewriter.create<FuncOp>(parentModule.getLoc(), func_name, FunctionType::get(parentModule.getContext(), Input, Output));
  mlir::MLIRContext *ctx = parentModule.getContext();
  llvm::SmallVector<mlir::Attribute, 4> attrs;
  attrs.push_back(mlir::ArrayAttr::get(ctx,
                                  {mlir::StringAttr::get(ctx, "prefer-vector-width"),
                                  mlir::StringAttr::get(ctx, "128")}));
  attrs.push_back(mlir::ArrayAttr::get( ctx, {mlir::StringAttr::get(ctx, "target-cpu"),
                                          mlir::StringAttr::get(ctx, "thunderx2t99")}));
  func_op->setAttr("passthrough", mlir::ArrayAttr::get(ctx, attrs));

  auto entry_block = func_op.addEntryBlock();

  // Build the dominance tree of the parent op of the block
  Region *region = block->getParent();
  Operation *parent_op = region->getParentOp();
  auto dom_info = mlir::DominanceInfo(parent_op);

  // std::set<int64_t> consts;
  llvm::SmallVector<Value, 1> vals;
  llvm::SmallVector<arith::ConstantOp, 1> consts;
  bool add_yield = false;

  // Walk the block and find out all the variables that were defined outside
  // this block and are used inside the block (i.e., all the variables x that
  // properly dominate the block). The only things we will redefine inside the
  // entry block are constants. For all other variables, we will add them as
  // inputs to the function
  block->walk([&](Operation *inst) {
    for (Value val: inst->getOperands()){
      if (dom_info.properlyDominates(val, parent_op)){
        if(func_name=="kernel_tail"){
        llvm::errs()<<"dominates?\n";

        }
        arith::ConstantOp const_op = val.getDefiningOp<arith::ConstantOp>();
        if (const_op) {
          // It's useless to add many times the same index
          if (std::find(consts.begin(), consts.end(), const_op) ==
              consts.end()) {
            consts.push_back(const_op);
            rewriter.setInsertionPointToStart(entry_block);
            Operation *new_const = rewriter.clone(*const_op);
            rewriter.replaceOpWithinBlock(const_op, new_const->getResult(0),
                                          block);
          }
        } else {
          if (std::find(vals.begin(), vals.end(), val) == vals.end()) {
            func_op.insertArgument(vals.size(), val.getType(), {});
            vals.push_back(val);
          }
        }
      }
    }

    // Remove Yield operations and signal to add it from the caller site
    // TODO: this is wrong if yield has results/operands connected to it.
    // We should clone the yield in the caller block
    if (dyn_cast<scf::YieldOp>(inst) && inst->getBlock() == block) {
      add_yield = true;
      rewriter.eraseOp(inst);
    }
  });

  llvm::SmallVector<Type, 1> newtypes;
  // We are not done yet. We need to merge the block into the entry block. To do
  // this: 1 If an operation in the block is using a value coming from the block
  //   argument, add the value as function argument and replace the value with
  //   it
  // 2 If an operation in the block is using a value generated outside the
  //   block, simply replace its value with a funciton argument

  // Step 1: get all the block arguments, add them as function arguments and
  // replece their use inside the block
  for (auto block_arg : block->getArguments()) {
    func_op.insertArgument(vals.size(), block_arg.getType(), {});
    auto arg = func_op.getArgument(vals.size());
    block_arg.replaceAllUsesWith(arg);
    newtypes.push_back(block_arg.getType());
  }

  // Step 2: replace all the values that are pointing outside the block and
  // replace them with function arguments
  auto args = func_op.getArguments();
  for (unsigned i = 0; i < vals.size(); i++) {
    auto val = vals[i];
    auto arg = args[i];
    val.replaceUsesWithIf(arg, [&](OpOperand &op) {
      return dom_info.dominates(block, op.getOwner()->getBlock());
    });
  }

  // Save some information about the original block. Once the block is merged
  // inside the entry block these information won't be available anymore
  bool has_no_successor = block->hasNoSuccessors();
  Block *succ = (has_no_successor ? nullptr : block->getSuccessor(0));

  // Remove all arguments from the block signature
  for (unsigned i = 0; i < block->getNumArguments(); i++) {
    block->eraseArgument(i);
  }

  // Merge block into entry_block (this destroys block)
  rewriter.mergeBlocks(block, entry_block);

  // Add a returnOp into the block to properly terminate it
  rewriter.setInsertionPointToEnd(entry_block);
  rewriter.create<ReturnOp>(loc);

  // We are done with the callee. Now we have to work on the caller. The overall
  // idea is to insert a new_block right before the successor of the old block.
  // If the old block has no successors, then add it at the end of the region
  Block *new_block = nullptr;
  if (has_no_successor) {
    new_block = rewriter.createBlock(region, region->end(), newtypes);
  } else {
    new_block = rewriter.createBlock(succ, newtypes);
  }

  // Remember to add the block arguments as inputs to the function
  for (unsigned i = 0; i < new_block->getNumArguments(); i++) {
    vals.push_back(new_block->getArgument(i));
  }

  // Create the call
  rewriter.create<CallOp>(loc, func_op, vals);

  if (add_yield) {
    rewriter.create<scf::YieldOp>(loc);
  }
}

// Walk the for loops and find the one that as operands. In GEMM is the
// micro-kernel.
// TODO: we should have the linalg::split to signal the microkernel of the
// operation and use it to run the function extractor if needed
struct ExtractKernelPass : public ExtractKernelPassBase<ExtractKernelPass> {

  ExtractKernelPass() = default;
  ExtractKernelPass(const ExtractKernelPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {}
  void runOnOperation() override {
    // Get the current FuncOp operation being operated on.
    auto module = getOperation();
    scf::ForOp loop;

    for (FuncOp func : module.getOps<FuncOp>()) {
      // Walk the operations within the function.
      func.walk([&](scf::ForOp forop) {
        if (forop.getNumIterOperands()) {
          loop = forop;
        }
      });
    }

    IRRewriter rewriter(module.getContext());
    extract_function("kernel", loop->getBlock(), module, rewriter,
                     module.getLoc());
  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>> mlir::createExtractKernelPass() {
  return std::make_unique<ExtractKernelPass>();
}

void extract_function_2(StringRef func_name, Block *block, ModuleOp parentModule, RewriterBase &rewriter, Block* origin, Location loc)
{
  SmallVector<Type, 1> Input, Output;

  // Create the function (callee site) with an empty block
  rewriter.setInsertionPointToStart(parentModule.getBody());
  auto func_op = rewriter.create<FuncOp>(parentModule.getLoc(), func_name, FunctionType::get(parentModule.getContext(), Input, Output));
  auto entry_block = func_op.addEntryBlock();

  // Build the dominance tree of the parent op of the block
  Region *region = block->getParent();
  Operation *parent_op = region->getParentOp();
  auto dom_info = mlir::DominanceInfo(parent_op);

  //std::set<int64_t> consts;
  llvm::SmallVector<Value, 1> vals;
  llvm::SmallVector<arith::ConstantOp, 1> consts;

  // Walk the block and find out all the variables that were defined outside 
  // this block and are used inside the block (i.e., all the variables x that
  // properly dominate the block). The only things we will redefine inside the
  // entry block are constants. For all other variables, we will add them as 
  // inputs to the function
  block->walk([&](Operation *inst) {
    for (Value val: inst->getOperands()){
      if (dom_info.properlyDominates(val, &block->getOperations().front())){
        arith::ConstantOp const_op = val.getDefiningOp<arith::ConstantOp>();
        if (const_op){
          // It's useless to add many times the same index
          if (std::find(consts.begin(), consts.end(), const_op) == consts.end()){
            consts.push_back(const_op);
          }
        } else  {
          if (std::find(vals.begin(), vals.end(), val) == vals.end()){
            func_op.insertArgument(vals.size(), val.getType(), {});
            vals.push_back(val);
          }
        }
      } 
    }
  });

  llvm::SmallVector<Type, 1> newtypes;
  // We are not done yet. We need to merge the block into the entry block. To do this:
  // 1 If an operation in the block is using a value coming from the block
  //   argument, add the value as function argument and replace the value with it
  // 2 If an operation in the block is using a value generated outside the
  //   block, simply replace its value with a funciton argument

  // Step 1: get all the block arguments, add them as function arguments and
  // replece their use inside the block
  //for (auto block_arg : block->getArguments())
  //{
  //  func_op.insertArgument(vals.size(), block_arg.getType(), {});
  //  auto arg = func_op.getArgument(vals.size());
  //  block_arg.replaceAllUsesWith(arg);
  //  newtypes.push_back(block_arg.getType());
  //}

  // Step 2: replace all the values that are pointing outside the block and
  // replace them with function arguments
  auto args = func_op.getArguments();
  for (unsigned i = 0; i < vals.size(); i++)
  {
    auto val = vals[i];
    auto arg = args[i];
    val.replaceUsesWithIf(arg, [&](OpOperand &op)
                          { 
                            Operation *target = op.getOwner();
                            for(Operation &op : block->getOperations()){
                              if (&op == target) return true;
                            }
                            return false;
                           });
  }

  // Remove all arguments from the block signature
  //for (unsigned i = 0; i<block->getNumArguments(); i++){
  //  block->eraseArgument(i);
  //}

  // Merge block into entry_block (this destroys block)

  // Add constants
  rewriter.mergeBlocks(block, entry_block);
  rewriter.setInsertionPointToStart(entry_block);
  for (Operation * c : consts){
    Operation * new_const = rewriter.clone(*c);
    rewriter.replaceOpWithinBlock(c, new_const->getResult(0), entry_block);
  }

  // Add a returnOp into the block to properly terminate it
  rewriter.setInsertionPointToEnd(entry_block);
  //rewriter.create<ReturnOp>(loc);

  // We are done with the callee. Now we have to work on the caller. The overall
  // idea is to insert a new_block right before the successor of the old block.
  // If the old block has no successors, then add it at the end of the region

  rewriter.setInsertionPointToEnd(origin);

  // Create the call 
  rewriter.create<CallOp>(loc, func_op, vals);
  rewriter.create<ReturnOp>(loc);
}

// Walk the for loops and find the one that as operands. In GEMM is the micro-kernel.
// TODO: we should have the linalg::split to signal the microkernel of the operation
// and use it to run the function extractor if needed
struct ExtractKernelTailPass : public ExtractKernelTailPassBase<ExtractKernelTailPass> {

  ExtractKernelTailPass() = default;
  ExtractKernelTailPass(const ExtractKernelTailPass& pass) { }
  void getDependentDialects(DialectRegistry &registry) const override {

  }
  void runOnOperation() override {
    // Get the current FuncOp operation being operated on.
    auto module = getOperation();
    scf::ForOp loop;

    for (FuncOp func : module.getOps<FuncOp>()) {
      // Walk the operations within the function.
      func.walk([&](scf::ForOp forop) {
        if (forop.getNumIterOperands()){
          loop = forop;
        }
      });
    }

    IRRewriter rewriter(module.getContext());
    Block *tail = rewriter.splitBlock(loop->getBlock(), Block::iterator(loop->getNextNode()));
    //rewriter.setInsertionPointToEnd(loop->getBlock());
    // Create the call 
    //rewriter.create<CallOp>(loc, func_op, vals);
    //rewriter.create<ReturnOp>(module.getLoc());
    extract_function_2("kernel_tail", tail, module, rewriter, loop->getBlock(), module.getLoc());
    
  }
};

std::unique_ptr<mlir::OperationPass<ModuleOp>> mlir::createExtractKernelTailPass() {
  return std::make_unique<ExtractKernelTailPass>();
}
