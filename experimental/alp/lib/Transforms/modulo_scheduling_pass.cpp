//===-- modulo_scheduling_pass.cpp - Implement modulo scheduling ------*- c++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "alp/Transforms/PassDetail.h"
#include "alp/Transforms/Passes.h"

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <unordered_map>

#define DEBUG_TYPE "modulo-scheduling"
using namespace mlir;

namespace {
enum StageType { Compute, Load };

struct ModuloSchedulingPass
    : public ModuloSchedulingPassBase<ModuloSchedulingPass> {
  // ModuloScheduling(int unrollFactor):unrollFactor_(unrollFactor){}
  ModuloSchedulingPass() = default;
  ModuloSchedulingPass(const ModuloSchedulingPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    LLVM_DEBUG(llvm::dbgs() << "modulo scheudling starts\n ");
    // Get the current FuncOp operation being operated on.
    FuncOp f = getOperation();

    scf::ForOp loop;

    //  Unroll the kernel
    f.walk([&](scf::ForOp forop) {
      if (forop.getNumIterOperands()) {
        loop = forop;
      }
    });

    if (loop) {
      assert(distance <= unrolling &&
             "pipeline distance cannot be bigger than unrolling");

      // Unroll
      (void)loopUnrollByFactor(loop, unrolling);

      /* Order/stage the instructions within the for loop.

      Initial loop (unroll==4) looks like:
      for (...){ L0 C0 L1 C1 L2 C2 }

      New loop should look like(distance==2):
      L0 L1 for (...){ C0 L2 C1 L3 C2 L0 C3 L1 } C0 C1

      */
      std::unordered_map<Operation *, int> stage_map;
      std::vector<std::vector<Operation *>> compute_queue(unrolling);
      std::vector<std::vector<Operation *>> load_queue(unrolling);

      StageType stage_type = Load;
      int current_compute_queue = 0;
      int current_load_queue = 0;

      // Take care of the stages
      for (Operation &operation : loop.getBody()->getOperations()) {
        Operation *op = &operation;
        if (dyn_cast<scf::YieldOp>(op)) {
          continue;
        }
        // This is a state machine with two states
        if (stage_type == Compute) {
          if (auto compute_op = dyn_cast<vector::OuterProductOp>(op)) {
            compute_queue[current_compute_queue].push_back(op);
          } else {
            stage_type = Load;
            current_compute_queue++;
            load_queue[current_load_queue].push_back(op);
          }
        } else { // if stage_type == Load
          if (auto compute_op = dyn_cast<vector::OuterProductOp>(op)) {
            stage_type = Compute;
            current_load_queue++;
            compute_queue[current_compute_queue].push_back(op);
          } else {
            load_queue[current_load_queue].push_back(op);
          }
        }
      }

      for (int i = 0; i < unrolling; i++) {
        for (Operation *op : load_queue[i]) {
          stage_map[op] = std::min(int(distance), i);
        }
        for (Operation *op : compute_queue[i]) {
          stage_map[op] = distance;
        }
      }

      mlir::scf::PipeliningOption options;
      options.getScheduleFn =
          [&](scf::ForOp forOp,
              std::vector<std::pair<Operation *, unsigned>> &schedule) {
            schedule.resize(forOp.getBody()->getOperations().size() - 1);

            int pos = 0;
            for (int i = 0; i < unrolling; i++) {
              auto current_comp_block = compute_queue[i];
              // Loads from the next block
              auto current_load_block = load_queue[(i + distance) % unrolling];

              // Get to the first load op (while scheduling the rest)
              unsigned load_start = 0;
              for (; load_start < current_load_block.size(); load_start++) {
                Operation *op = current_load_block[load_start];
                if (!dyn_cast<vector::LoadOp>(op)) {
                  schedule[pos++] = {op, stage_map[op]};
                } else {
                  break;
                }
              }
              size_t real_load_size = current_load_block.size() - load_start;
              unsigned min_size = 0;
              if (interleave) {
                min_size = std::min(current_comp_block.size(), real_load_size);
              }
              for (unsigned j = 0; j < min_size; j++) {
                Operation *load_op = current_load_block[j + load_start];
                Operation *compute_op = current_comp_block[j];
                schedule[pos++] = {compute_op, stage_map[compute_op]};
                schedule[pos++] = {load_op, stage_map[load_op]};
              }

              if (real_load_size > min_size) {
                for (unsigned j = min_size; j < real_load_size; j++) {
                  Operation *load_op = current_load_block[j + load_start];
                  schedule[pos++] = {load_op, stage_map[load_op]};
                }
              }

              if (current_comp_block.size() > min_size) {
                for (unsigned j = min_size; j < current_comp_block.size();
                     j++) {
                  Operation *compute_op = current_comp_block[j];
                  schedule[pos++] = {compute_op, stage_map[compute_op]};
                }
              }
            }
          };
      RewritePatternSet patterns(&getContext());
      // TODO: we should upstream this
      // options.dynamic_loops =
      // scf::PipelineDynamicLoops::EnableLoopVersioning;
      scf::populateSCFLoopPipeliningPatterns(patterns, options);
      (void)applyOpPatternsAndFold(loop, std::move(patterns));
    }
  }
  int unrollFactor_;
};
} // namespace

std::unique_ptr<mlir::OperationPass<FuncOp>>
mlir::createModuloSchedulingPass() {
  return std::make_unique<ModuloSchedulingPass>();
}
