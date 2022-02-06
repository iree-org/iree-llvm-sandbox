//===- LinalgTensorCodegenDriver.cpp - Linalg transformation driver--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgExt/LinalgExtBufferization.h"
#include "Dialects/LinalgExt/LinalgExtDialect.h"
#include "Transforms/PassDetail.h"
#include "Transforms/Passes.h"
#include "Transforms/Transforms.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/LinalgInterfaceImpl.h"
#include "mlir/Dialect/Linalg/ComprehensiveBufferize/ModuleBufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::linalg;

namespace {

static void
getAtMostNEnclosingLoops(Operation *op, int64_t nLoops,
                         SmallVector<scf::ForOp> &reverseEnclosingLoops) {
  scf::ForOp outermostEnclosingForOp = nullptr;
  Operation *nextEnclosingOp = op->getParentOp();
  while (nLoops-- > 0 &&
         (outermostEnclosingForOp = dyn_cast<scf::ForOp>(nextEnclosingOp))) {
    reverseEnclosingLoops.push_back(outermostEnclosingForOp);
    nextEnclosingOp = outermostEnclosingForOp->getParentOp();
  }
}

struct LinalgFusePass : public LinalgFuseBase<LinalgFusePass> {
  LinalgFusePass() = default;
  LinalgFusePass(const LinalgFusePass &pass) {}
  void runOnOperation() override;
};

struct LinalgFuseOutputIntoReductionPass
    : public LinalgFuseOutputIntoReductionBase<
          LinalgFuseOutputIntoReductionPass> {
  LinalgFuseOutputIntoReductionPass() = default;
  LinalgFuseOutputIntoReductionPass(
      const LinalgFuseOutputIntoReductionPass &pass) {}
  void runOnOperation() override;
};

struct LinalgSingleTilingExpertPass
    : public LinalgSingleTilingExpertBase<LinalgSingleTilingExpertPass> {
  LinalgSingleTilingExpertPass() = default;
  LinalgSingleTilingExpertPass(const LinalgSingleTilingExpertPass &pass) {}

  /// Function pass entry point.
  void runOnOperation() override;
};

struct LinalgBufferizationDriverPass
    : public LinalgBufferizationDriverBase<LinalgBufferizationDriverPass> {
  LinalgBufferizationDriverPass() = default;
  LinalgBufferizationDriverPass(const LinalgBufferizationDriverPass &pass) {}

  void runOnOperation() override;
};

struct LinalgVectorLoweringPass
    : public LinalgVectorLoweringBase<LinalgVectorLoweringPass> {
  LinalgVectorLoweringPass(int64_t vectorLoweringStage = 0) {
    this->vectorLoweringStage.setValue(vectorLoweringStage);
  }
  LinalgVectorLoweringPass(const LinalgVectorLoweringPass &pass) {
    this->vectorLoweringStage.setValue(pass.vectorLoweringStage);
  }

  void runOnOperation() override;
};

struct LLVMLoweringPass : public LLVMLoweringBase<LLVMLoweringPass> {
  LLVMLoweringPass() = default;
  LLVMLoweringPass(const LLVMLoweringPass &pass) {}

  void runOnOperation() override;
};

struct UnrollOneVectorOpPass
    : public UnrollOneVectorOpBase<UnrollOneVectorOpPass> {
  UnrollOneVectorOpPass() = default;
  UnrollOneVectorOpPass(const UnrollOneVectorOpPass &pass) {}
  void runOnOperation() override;
};

struct UnrollOneParentLoopPass
    : public UnrollOneParentLoopBase<UnrollOneParentLoopPass> {
  UnrollOneParentLoopPass() = default;
  UnrollOneParentLoopPass(const UnrollOneParentLoopPass &pass) {}
  void runOnOperation() override;
};

struct OutlineOneParentLoopPass
    : public OutlineOneParentLoopBase<OutlineOneParentLoopPass> {
  OutlineOneParentLoopPass() = default;
  OutlineOneParentLoopPass(const OutlineOneParentLoopPass &pass) {}
  void runOnOperation() override;
};

struct PipelineOneParentLoopPass
    : public PipelineOneParentLoopBase<PipelineOneParentLoopPass> {
  PipelineOneParentLoopPass() = default;
  PipelineOneParentLoopPass(const PipelineOneParentLoopPass &pass) {}
  void runOnOperation() override;
};

} // namespace

void LLVMLoweringPass::runOnOperation() {
  OpPassManager dynamicPM(ModuleOp::getOperationName());
  // This is a failsafe catchall, if it does something performance opportunities
  // have been missed previously.
  dynamicPM.addNestedPass<FuncOp>(createConvertVectorToSCFPass());
  dynamicPM.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  dynamicPM.addPass(createAsyncToAsyncRuntimePass());
  dynamicPM.addPass(createAsyncRuntimeRefCountingPass());
  dynamicPM.addPass(createAsyncRuntimeRefCountingOptPass());
  dynamicPM.addPass(createCanonicalizerPass());
  dynamicPM.addPass(createLowerAffinePass());
  dynamicPM.addPass(createLowerToCFGPass());
  dynamicPM.addPass(createConvertLinalgToLLVMPass());
  dynamicPM.addPass(createConvertVectorToLLVMPass(
      // clang-format off
      LowerVectorToLLVMOptions()
        .enableReassociateFPReductions(reassociateFPReductions)
        .enableIndexOptimizations(indexOptimizations)
        .enableArmNeon(armNeon)
        .enableArmSVE(armSVE)
        .enableAMX(amx)
        .enableX86Vector(x86Vector)));
  // clang-format on
  dynamicPM.addNestedPass<FuncOp>(createConvertMathToLLVMPass());
  dynamicPM.addPass(createMemRefToLLVMPass());
  dynamicPM.addPass(createConvertAsyncToLLVMPass());
  dynamicPM.addPass(createLowerToLLVMPass());
  dynamicPM.addPass(createCanonicalizerPass());
  dynamicPM.addPass(createCSEPass());
  if (failed(runPipeline(dynamicPM, getOperation())))
    return signalPassFailure();

  // Make all arguments noalias for now.
  getOperation().walk([](LLVM::LLVMFuncOp funcOp) {
    for (int64_t i = 0; i < funcOp.getNumArguments(); ++i) {
      if (!funcOp.getType().getParamType(i).isa<LLVM::LLVMPointerType>())
        continue;
      funcOp.setArgAttr(i, "llvm.noalias", UnitAttr::get(funcOp.getContext()));
    }
  });
}

/// Return the neutral element as a new Value.
/// For now, just assume it is the zero of type.
/// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<arith::ConstantOp>(op.getOwner()->getLoc(), t,
                                     b.getZeroAttr(t));
}

/// Collect all Linalg ops, they must all have tensor semantics.
/// For now this just fuses everything.
// TODO: finer control.
void LinalgFusePass::runOnOperation() {
  FuncOp funcOp = getOperation();
  if (anchorOpName.empty())
    return;

  // Set up tiling and vectorization options.
  LinalgTilingAndFusionOptions tilingOptions;
  tilingOptions.tileSizes = {tileSizes.begin(), tileSizes.end()};
  tilingOptions.tileInterchange = {tileInterchange.begin(),
                                   tileInterchange.end()};

  // Set up padding options.
  // TODO: Replace the lambdas by either functions defined in MLIR core or even
  // adapt the LinalgPaddingOptions to take the `hoistPaddings` and
  // `packPaddings` arrays directly.
  auto packFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < packPaddings.size()
               ? packPaddings[opOperand.getOperandNumber()]
               : false;
  };
  auto hoistingFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < hoistPaddings.size()
               ? hoistPaddings[opOperand.getOperandNumber()]
               : 0;
  };
  auto transposeFunc = [&](OpOperand &opOperand) {
    SmallVector<int64_t> transposeVector = {};
    if (opOperand.getOperandNumber() >= transposePaddings.size())
      return transposeVector;
    SmallVector<StringRef> elems;
    StringRef(transposePaddings[opOperand.getOperandNumber()])
        .split(elems, ':');
    for (StringRef elem : elems)
      transposeVector.push_back(std::stoi(elem.str()));
    return transposeVector;
  };
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);
  paddingOptions.setPaddingTransposeComputationFunction(transposeFunc);

  CodegenStrategy strategy;
  strategy.tileAndFuseIf(!tileSizes.empty(), anchorOpName, tilingOptions)
      .padIf(pad, "", paddingOptions)
      .vectorizeIf(vectorize, "", nullptr, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());

  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}

void LinalgFuseOutputIntoReductionPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  if (funcOp.getName() != anchorFuncOpName)
    return;

  mlir::RewritePatternSet patterns(funcOp.getContext());
  populateFuseFillIntoReductionPatterns(patterns);
  (void)mlir::applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
}

void LinalgSingleTilingExpertPass::runOnOperation() {
  FuncOp funcOp = getOperation();

  // Set up tiling and vectorization options.
  LinalgTilingOptions tilingOptions;
  bool doTiling = false;
  if (!tileSizes.empty()) {
    doTiling = true;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
  }
  if (!tileInterchange.empty())
    tilingOptions = tilingOptions.setInterchange(
        SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));
  if (scalarizeDynamicDims) {
    doTiling = true;
    tilingOptions = tilingOptions.scalarizeDynamicDims();
  }
  tilingOptions = tilingOptions.setPeeledLoops(peeledLoops);

  // Set up padding options.
  // TODO: Replace the lambdas by either functions defined in MLIR core or even
  // adapt the LinalgPaddingOptions to take the `hoistPaddings` and
  // `packPaddings` arrays directly.
  auto packFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < packPaddings.size()
               ? packPaddings[opOperand.getOperandNumber()]
               : false;
  };
  auto hoistingFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < hoistPaddings.size()
               ? hoistPaddings[opOperand.getOperandNumber()]
               : 0;
  };
  auto transposeFunc = [&](OpOperand &opOperand) {
    SmallVector<int64_t> transposeVector = {};
    if (opOperand.getOperandNumber() >= transposePaddings.size())
      return transposeVector;
    SmallVector<StringRef> elems;
    StringRef(transposePaddings[opOperand.getOperandNumber()])
        .split(elems, ':');
    for (StringRef elem : elems)
      transposeVector.push_back(std::stoi(elem.str()));
    return transposeVector;
  };
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);
  paddingOptions.setPaddingTransposeComputationFunction(transposeFunc);

  auto vectorizeFilter = [&](mlir::Operation *op) {
    return success(!vectorizeOnlyTiled || op->getParentOfType<scf::ForOp>());
  };
  CodegenStrategy strategy;
  StringRef genericOpName = GenericOp::getOperationName();
  strategy.tileIf(doTiling, anchorOpName, tilingOptions)
      .padIf(pad, anchorOpName, paddingOptions)
      .decomposeIf(decomposeToLowerDimOp)
      .generalizeIf(generalize, anchorOpName)
      .interchangeIf(!iteratorInterchange.empty(), iteratorInterchange)
      .vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName,
                   vectorizeFilter, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}

void LinalgBufferizationDriverPass::runOnOperation() {
  OpPassManager dynamicPM(ModuleOp::getOperationName());
  dynamicPM.addPass(createCanonicalizerPass());
  dynamicPM.addPass(createCSEPass());
  dynamicPM.addPass(
      createLinalgComprehensiveModuleBufferizePass(/*useLinalgCopy=*/true));
  if (failed(runPipeline(dynamicPM, getOperation())))
    return signalPassFailure();
  // Perform buffer-level hoistings.
  getOperation().walk(
      [&](FuncOp funcOp) { hoistRedundantVectorTransfers(funcOp); });
}

void LinalgVectorLoweringPass::runOnOperation() {
  vector::VectorTransposeLowering vectorTransposeLowering =
      llvm::StringSwitch<vector::VectorTransposeLowering>(
          lowerVectorTransposeTo.getValue())
          .Case("eltwise", vector::VectorTransposeLowering::EltWise)
          .Case("flat_transpose", vector::VectorTransposeLowering::Flat)
          .Case("shuffle", vector::VectorTransposeLowering::Shuffle)
          .Default(vector::VectorTransposeLowering::EltWise);
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
  vector::VectorTransformsOptions vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransposeLowering(vectorTransposeLowering)
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);
  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(unrollVectorTransfers)
          .enableLowerPermutationMaps();

  LinalgVectorLoweringOptions vectorLoweringOptions =
      LinalgVectorLoweringOptions()
          // Lowering of vector contractions.
          .enableContractionLowering(vectorLoweringStage >= 0)
          // Lowering of vector multi_reduction.
          .enableMultiReductionLowering(vectorLoweringStage >= 1)
          // Whether to split full/partial vector.transfer ops.
          .enableTransferPartialRewrite(vectorLoweringStage >= 2 &&
                                        vectorTransferSplit !=
                                            vector::VectorTransferSplit::None)
          // Set the maximum vector load / store rank.
          .setMaxTransferRank(maxTransferRank)
          // Lower vector.transfer to vector.transfer of max rank.
          .enableTransferLowering(vectorLoweringStage >= 3)
          // Conversion to scf.
          .enableTransferToSCFConversion(vectorLoweringStage >= 4)
          .setVectorTransferToSCFOptions(vectorTransferToSCFOptions)
          // Lowering of vector.shape_cast.
          .enableShapeCastLowering(vectorLoweringStage >= 5)
          // Lowering of vector.transpose.
          .enableVectorTransposeLowering(vectorLoweringStage >= 6)
          .setVectorTransformsOptions(vectorTransformOptions)
          .enableAVX2Lowering(lowerVectorTransposeToAVX2)
          .setAVX2LoweringOptions(
              x86vector::avx2::LoweringOptions().setTransposeOptions(
                  x86vector::avx2::TransposeLoweringOptions()
                      .lower4x8xf32(lowerVectorTransposeToAVX2)
                      .lower8x8xf32(lowerVectorTransposeToAVX2)));

  CodegenStrategy strategy;
  strategy.vectorLowering(vectorLoweringOptions);
  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  FuncOp funcOp = getOperation();
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}

void UnrollOneVectorOpPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName)
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  vector::populateVectorUnrollPatterns(
      patterns, vector::UnrollVectorOptions()
                    .setNativeShape(targetShape)
                    .setFilterConstraint([&](Operation *op) {
                      auto unrollInterface =
                          dyn_cast<VectorUnrollOpInterface>(op);
                      if (!unrollInterface ||
                          op->getName().getStringRef() != anchorOpName ||
                          !sourceShape.hasValue() ||
                          !unrollInterface.getShapeForUnroll().hasValue())
                        return failure();

                      ArrayRef<int64_t> sourceShapeToMatch{sourceShape};
                      auto shapeForUnroll =
                          unrollInterface.getShapeForUnroll().getValue();
                      ArrayRef<int64_t> actualSourceShape{
                          shapeForUnroll.begin(), shapeForUnroll.end()};
                      return success(sourceShapeToMatch == actualSourceShape);
                    }));
  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

void UnrollOneParentLoopPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName)
    return;

  // Poor man's op targeting.
  getOperation().walk([&](Operation *op) {
    if (op->getName().getStringRef() != anchorOpName)
      return WalkResult::advance();
    SmallVector<scf::ForOp> reverseEnclosingLoops;
    getAtMostNEnclosingLoops(op, parentLoopNum, reverseEnclosingLoops);
    if (failed(loopUnrollByFactor(reverseEnclosingLoops.back(), unrollFactor)))
      signalPassFailure();
    return WalkResult::interrupt();
  });
}

scf::ExecuteRegionOp outlineInExecuteRegion(RewriterBase &b, Operation *op) {
  if (op->getNumRegions() != 1)
    return nullptr;
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  scf::ExecuteRegionOp executeRegionOp =
      b.create<scf::ExecuteRegionOp>(op->getLoc(), op->getResultTypes());
  {
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToStart(&executeRegionOp.getRegion().emplaceBlock());
    Operation *clonedOp = b.cloneWithoutRegions(*op);
    Region &clonedRegion = clonedOp->getRegions().front();
    assert(clonedRegion.empty() && "expected empty region");
    b.inlineRegionBefore(op->getRegions().front(), clonedRegion,
                         clonedRegion.end());
    b.create<scf::YieldOp>(op->getLoc(), clonedOp->getResults());
  }
  b.replaceOp(op, executeRegionOp.getResults());
  return executeRegionOp;
}

void OutlineOneParentLoopPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName)
    return;

  // Poor man's op targeting.
  getOperation().walk([&](Operation *op) {
    if (op->getName().getStringRef() != anchorOpName)
      return WalkResult::advance();
    SmallVector<scf::ForOp> reverseEnclosingLoops;
    getAtMostNEnclosingLoops(op, parentLoopNum, reverseEnclosingLoops);
    IRRewriter b(op->getContext());
    scf::ExecuteRegionOp exec =
        outlineInExecuteRegion(b, reverseEnclosingLoops.back());
    if (failed(outlineSingleBlockRegion(b, op->getLoc(), exec.getRegion(),
                                        resultFuncName)))
      signalPassFailure();
    return WalkResult::interrupt();
  });
}

// Naive schedule: Schedule ops as early as possible.
static void
loopScheduling(scf::ForOp forOp,
               std::vector<std::pair<Operation *, unsigned>> &schedule,
               unsigned II, unsigned readLatency) {
  auto getLatency = [&](Operation *op) {
    if (isa<vector::TransferReadOp>(op))
      return readLatency;
    return unsigned(1);
  };

  DenseMap<Operation *, unsigned> opCycles;
  std::map<unsigned, std::vector<Operation *>> wrappedSchedule;
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (isa<scf::YieldOp>(op))
      continue;
    unsigned earlyCycle = 0;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      earlyCycle = std::max(earlyCycle, opCycles[def] + getLatency(def));
    }
    opCycles[&op] = earlyCycle;
    wrappedSchedule[earlyCycle % II].push_back(&op);
  }
  for (auto it : wrappedSchedule) {
    for (Operation *op : it.second) {
      unsigned cycle = opCycles[op];
      schedule.push_back(std::make_pair(op, cycle / II));
    }
  }
}

void PipelineOneParentLoopPass::runOnOperation() {
  if (getOperation().getName() != anchorFuncOpName)
    return;

  // Poor man's op targeting.
  getOperation().walk([&](Operation *op) {
    if (op->getName().getStringRef() != anchorOpName)
      return WalkResult::advance();
    SmallVector<scf::ForOp> reverseEnclosingLoops;
    getAtMostNEnclosingLoops(op, parentLoopNum, reverseEnclosingLoops);

    scf::ForOp loopToPipeline = reverseEnclosingLoops.back();
    scf::PipeliningOption schedule;
    schedule.getScheduleFn =
        [&](scf::ForOp forOp,
            std::vector<std::pair<Operation *, unsigned>> &order) {
          if (forOp != loopToPipeline)
            return;
          return loopScheduling(forOp, order, II, readLatency);
        };
    RewritePatternSet patterns(op->getContext());
    scf::populateSCFLoopPipeliningPatterns(patterns, schedule);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
    return WalkResult::interrupt();
  });
}

//===----------------------------------------------------------------------===//
// Pass creation entry points.
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgFusePass() {
  return std::make_unique<LinalgFusePass>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgFuseOutputIntoReductionPass() {
  return std::make_unique<LinalgFuseOutputIntoReductionPass>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgSingleTilingExpertPass() {
  return std::make_unique<LinalgSingleTilingExpertPass>();
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createLinalgBufferizationDriverPass() {
  return std::make_unique<LinalgBufferizationDriverPass>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgVectorLoweringPass(int64_t vectorLoweringStage) {
  return std::make_unique<LinalgVectorLoweringPass>(vectorLoweringStage);
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createLLVMLoweringPass() {
  return std::make_unique<LLVMLoweringPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createUnrollOneVectorOpPass() {
  return std::make_unique<UnrollOneVectorOpPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createUnrollOneParentLoopPass() {
  return std::make_unique<UnrollOneParentLoopPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createOutlineOneParentLoopPass() {
  return std::make_unique<OutlineOneParentLoopPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createPipelineOneParentLoopPass() {
  return std::make_unique<PipelineOneParentLoopPass>();
}

//===----------------------------------------------------------------------===//
// Transforms
//===----------------------------------------------------------------------===//

void mlir::addLowerToVectorTransforms(OpPassManager &passManager) {
  passManager.addPass(createLinalgVectorLoweringPass(0));
  passManager.addPass(createLinalgVectorLoweringPass(1));
  passManager.addPass(createLinalgVectorLoweringPass(2));
  passManager.addPass(createLinalgVectorLoweringPass(3));
  passManager.addPass(createLinalgVectorLoweringPass(4));
  passManager.addPass(createLinalgVectorLoweringPass(5));
  passManager.addPass(createLinalgVectorLoweringPass(6));
}
