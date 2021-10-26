//===- LinalgTensorCodegenDriver.cpp - Linalg transformation driver--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/HoistPadding.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {
struct LinalgTensorCodegenDriverPass
    : public LinalgTensorCodegenDriverBase<LinalgTensorCodegenDriverPass> {
  LinalgTensorCodegenDriverPass() = default;
  LinalgTensorCodegenDriverPass(const LinalgTensorCodegenDriverPass &pass) {}

  /// Function pass entry point.
  void runOnOperation() override;

 private:
  void fuseAll(FuncOp funcOp);
  void runOpAnchoredStrategy(FuncOp funcOp);
  void runComprehensiveBufferization();
  void runVectorLowering();
  void runLowerToLLVM();
};
}  // namespace

void LinalgTensorCodegenDriverPass::runLowerToLLVM() {
  OpPassManager dynamicPM("builtin.module");
  OpPassManager &nestedFuncOpPM = dynamicPM.nest<FuncOp>();
  // This is a failsafe catchall, if it does something performance opportunities
  // have been missed previously.
  nestedFuncOpPM.addPass(createConvertVectorToSCFPass());
  nestedFuncOpPM.addPass(createConvertLinalgToLoopsPass());
  dynamicPM.addPass(createCanonicalizerPass());
  dynamicPM.addPass(createLowerAffinePass());
  dynamicPM.addPass(createLowerToCFGPass());
  dynamicPM.addPass(createConvertLinalgToLLVMPass());
  dynamicPM.addPass(createConvertVectorToLLVMPass());
  dynamicPM.addPass(createMemRefToLLVMPass());
  dynamicPM.addPass(createLowerToLLVMPass());
  dynamicPM.addPass(createCanonicalizerPass());
  dynamicPM.addPass(createCSEPass());
  if (failed(runPipeline(dynamicPM, getOperation())))
    return signalPassFailure();
}

/// Return the neutral element as a new Value.
/// For now, just assume it is the zero of type.
/// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<ConstantOp>(op.getOwner()->getLoc(), t, b.getZeroAttr(t));
}

/// Collect all Linalg ops, they must all have tensor semantics.
/// For now this just fuses everything.
// TODO: finer control.
void LinalgTensorCodegenDriverPass::fuseAll(FuncOp funcOp) {
  SmallVector<LinalgOp> linalgOps;
  auto walkResult = funcOp.walk([&](LinalgOp op) {
    if (!op.hasTensorSemantics()) return WalkResult::interrupt();
    linalgOps.push_back(op);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) return signalPassFailure();

  // Compute the tile sizes and the interchange.
  LinalgOp rootOp = linalgOps.back();
  assert(tileSizes.size() >= rootOp.getNumLoops() &&
         "expect one tile sizes per root op loop dimension");
  assert(tileInterchange.empty() ||
         tileInterchange.size() == tileSizes.size() &&
             "expect the number of tile sizes and interchange dims to match");
  SmallVector<int64_t> rootTileSizes(tileSizes.begin(),
                                     tileSizes.begin() + rootOp.getNumLoops());
  SmallVector<int64_t> rootInterchange =
      tileInterchange.empty()
          ? llvm::to_vector<6>(llvm::seq<int64_t>(0, rootOp.getNumLoops()))
          : SmallVector<int64_t>(
                tileInterchange.begin(),
                tileInterchange.begin() + rootOp.getNumLoops());

  // Tile the root operation and fuse it with its producers.
  OpBuilder b(funcOp.getContext());
  FailureOr<TileLoopNest> tileLoopNest =
      tileConsumerAndFuseProducers(b, rootOp, rootTileSizes, rootInterchange);
  if (failed(tileLoopNest)) return signalPassFailure();
  rootOp->replaceAllUsesWith(tileLoopNest->getRootOpReplacementResults());
}

void LinalgTensorCodegenDriverPass::runOpAnchoredStrategy(FuncOp funcOp) {
  if (anchorOpName.empty()) return;

  if (fuse) return fuseAll(funcOp);

  // Set up tiling and vectorization options.
  LinalgTilingOptions tilingOptions;
  if (!tileSizes.empty()) tilingOptions = tilingOptions.setTileSizes(tileSizes);
  if (!tileInterchange.empty())
    tilingOptions = tilingOptions.setInterchange(
        SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));
  if (scalarizeDynamicDims)
    tilingOptions = tilingOptions.scalarizeDynamicDims();
  tilingOptions = tilingOptions.setPeeledLoops(peeledLoops);
  if (pad) {
    auto nofoldFunc = [&](OpOperand &opOperand) {
      return llvm::count(nofoldOperands, opOperand.getOperandNumber()) != 0;
    };
    tilingOptions =
        tilingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
    tilingOptions =
        tilingOptions.setPaddingNoFoldComputationFunction(nofoldFunc);
  }
  CodegenStrategy strategy;
  StringRef genericOpName = GenericOp::getOperationName();
  strategy
      .tileIf(!tileSizes.empty() || scalarizeDynamicDims, anchorOpName,
              tilingOptions)
      .generalizeIf(generalize, anchorOpName)
      .interchangeIf(!iteratorInterchange.empty(), iteratorInterchange)
      .vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM("builtin.func");
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();
}

void LinalgTensorCodegenDriverPass::runComprehensiveBufferization() {
  OpPassManager dynamicPM("builtin.module");
  dynamicPM.addPass(createCanonicalizerPass());
  dynamicPM.addPass(createCSEPass());
  dynamicPM.addPass(createLinalgComprehensiveModuleBufferizePass());
  if (failed(runPipeline(dynamicPM, getOperation())))
    return signalPassFailure();
}

void LinalgTensorCodegenDriverPass::runVectorLowering() {
  vector::VectorTransposeLowering vectorTransposeLowering =
      llvm::StringSwitch<vector::VectorTransposeLowering>(
          lowerVectorTransposeTo.getValue())
          .Case("eltwise", vector::VectorTransposeLowering::EltWise)
          .Default(vector::VectorTransposeLowering::Flat);
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      llvm::StringSwitch<vector::VectorMultiReductionLowering>(
          lowerVectorMultiReductionTo.getValue())
          .Case("innerreduction",
                vector::VectorMultiReductionLowering::InnerReduction)
          .Default(vector::VectorMultiReductionLowering::InnerParallel);
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          lowerVectorContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // Per-function lowering pipeline.
  getOperation().walk([&](FuncOp funcOp) {
    CodegenStrategy strategy;
    strategy.vectorLowering(
        LinalgVectorLoweringOptions()
            // Set the maximum vector load / store rank.
            .enableTransferLowering()
            .setMaxTransferRank(maxTransferRank)
            // Lowering of vector transpose.
            .enableVectorTransposeLowering(true)
            // Lowering of vector contractions.
            .enableContractionLowering()
            // Lowering of vector multi_reduction.
            .enableMultiReductionLowering()
            // Whether to split full/partial vector.transfer ops.
            .enableTransferPartialRewrite()
            .enableTransferPartialRewrite(vectorTransferSplit !=
                                          vector::VectorTransferSplit::None)
            .setVectorTransformsOptions(
                vector::VectorTransformsOptions()
                    .setVectorTransposeLowering(vectorTransposeLowering)
                    .setVectorTransformsOptions(vectorContractLowering)
                    .setVectorMultiReductionLowering(
                        vectorMultiReductionLowering)
                    .setVectorTransferSplit(vectorTransferSplit))
            // Conversion to scf.
            .enableTransferToSCFConversion()
            .setVectorTransferToSCFOptions(
                VectorTransferToSCFOptions()
                    .enableFullUnroll(unrollVectorTransfers)
                    .enableLowerPermutationMaps()));
    // Created a nested OpPassManager and run.
    OpPassManager dynamicPM("builtin.func");
    strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
    if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();
  });
}

void LinalgTensorCodegenDriverPass::runOnOperation() {
  if (!anchorFuncOpName.empty()) {
    getOperation().walk([&](FuncOp funcOp) {
      if (funcOp.getName() != anchorFuncOpName) return;

      // Run transforms that require anchoring on a particular op. This only
      // applies if !anchorOpName.empty().
      runOpAnchoredStrategy(funcOp);

      // Run other transforms that do not require a named linalg op.
      // TODO: Move to codegen strategy as late transformations.
      if (!hoistPadding.empty()) {
        SmallVector<PadTensorOp> ops;
        funcOp.walk([&](PadTensorOp op) { ops.push_back(op); });
        for (auto it : llvm::reverse(llvm::zip(ops, hoistPadding)))
          (void)hoistPaddingOnTensors(std::get<0>(it), std::get<1>(it));
      }
      if (vectorizePadding) {
        OwningRewritePatternList extraVectorizationPatterns(
            funcOp.getContext());
        populatePadTensorOpVectorizationPatterns(extraVectorizationPatterns);
        (void)applyPatternsAndFoldGreedily(
            funcOp, std::move(extraVectorizationPatterns));
      }
    });
  }

  if (bufferize) {
    runComprehensiveBufferization();
    // Perform buffer-level hoistings.
    getOperation().walk(
        [&](FuncOp funcOp) { hoistRedundantVectorTransfers(funcOp); });
  }

  if (vectorLowering) runVectorLowering();

  if (llvmLowering) runLowerToLLVM();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinalgTensorCodegenDriverPass() {
  return std::make_unique<LinalgTensorCodegenDriverPass>();
}
