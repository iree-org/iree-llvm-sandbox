//===-- modulo_scheduling_pass.cpp - Implement modulo scheduling ------*- c++ -*-===//
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

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"

using namespace mlir;

namespace {
struct ModuloSchedulingPass
    : public ModuloSchedulingPassBase<ModuloSchedulingPass> {
  // ModuloScheduling(int unrollFactor):unrollFactor_(unrollFactor){}
  ModuloSchedulingPass() = default;
  ModuloSchedulingPass(const ModuloSchedulingPass &pass) {}
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect>();
  }
  void runOnFunction() override {
    // Get the current FuncOp operation being operated on.
    auto f = getFunction();

    scf::ForOp loop;

    //  Unroll the kernel
    f.walk([&](scf::ForOp forop) {
      if (forop.getNumIterOperands()) {
        loop = forop;
      }
    });

    if (loop) {
      // Unroll
      auto annotateFn = [this](unsigned i, Operation *op, OpBuilder b) {
        op->setAttr("unrolled_iteration", b.getUI32IntegerAttr(i));
      };

      (void)loopUnrollByFactor(loop, 2, annotateFn);

      // Pipeline the kernel
      RewritePatternSet patterns(&getContext());
      mlir::scf::PipeliningOption options;

      // Order/stage the instruction within the for loop. We are looking for a
      // pattern like %x0 = load -> (stage0, pos3) %y0 = load -> (stage0, pos4)
      // %z0 = outerprod(%x0, %y0) -> (stage1, pos2)
      // %x1 = load -> (stage1, pos0)
      // %y1 = load -> (stage1, pos1)
      // %z1 = outerprod(%x1, %y1) -> (stage1, pos5)
      std::unordered_map<Operation *, int> stage_map;
      std::unordered_map<Operation *, int> clock_map;

      int anchor0 = -1;
      int anchor1 = -1;
      int stage = 0;
      int clock = 0;
      // Take care of the stages
      for (Operation &operation : loop.getBody()->getOperations()) {
        Operation *op = &operation;
        if (dyn_cast<scf::YieldOp>(op)) {
          continue;
        }
        if (anchor0 == -1 && dyn_cast<vector::OuterProductOp>(op)) {
          anchor0 = clock;
          stage = 1;
        } else if (anchor1 == -1 && dyn_cast<vector::OuterProductOp>(op)) {
          anchor1 = clock;
        }
        clock_map[op] = clock++;
        stage_map[op] = stage;
      }

      // Take care of the clocks
      int diff = anchor1 - anchor0;
      for (Operation &operation : loop.getBody()->getOperations()) {
        Operation *op = &operation;
        int clock = clock_map[op];
        // swap the loads
        if (dyn_cast<scf::YieldOp>(op)) {
          continue;
        }
        if (clock < anchor0) {
          clock_map[op] = diff + clock;
        } else if (clock == anchor0) {
          clock_map[op] = diff - 1;
        } else if (clock > anchor0 && clock < anchor1) {
          clock_map[op] = clock - anchor0 - 1;
        }
      }

      options.getScheduleFn =
          [&](scf::ForOp forOp,
              std::vector<std::pair<Operation *, unsigned>> &schedule) {
            schedule.resize(forOp.getBody()->getOperations().size() - 1);
            for (auto p : clock_map) {
              Operation *op = p.first;
              int clock = p.second;
              int stage = stage_map[op];
              schedule[clock] = {op, stage};
            }
          };

      scf::populateSCFLoopPipeliningPatterns(patterns, options);
      (void)applyOpPatternsAndFold(loop, std::move(patterns));
    }
  }
  int unrollFactor_;
};
} // namespace

std::unique_ptr<mlir::FunctionPass> mlir::createModuloSchedulingPass() {
  return std::make_unique<ModuloSchedulingPass>();
}
