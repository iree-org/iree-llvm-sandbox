//===-- jasc_transform_ops.cc - Transform ops for Jasc dialect --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "jasc_transform_ops.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"

#define GET_OP_CLASSES
#include "jasc_transform_ops.cpp.inc"

//===----------------------------------------------------------------------===//
// MatchTagOp
//===----------------------------------------------------------------------===//

mlir::DiagnosedSilenceableFailure jasc::MatchTagOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  llvm::SmallVector<mlir::Operation *> matched_ops;
  for (mlir::Operation *op : state.getPayloadOps(getTarget())) {
    auto tags = op->getAttrOfType<mlir::ArrayAttr>("jasc_tags");
    if (tags == nullptr) continue;
    if (tags.size() < getTags().size()) continue;
    bool is_match = true;
    for (int i = 0; i < getTags().size(); i++) {
      if (tags[i] != getTags()[i]) {
        is_match = false;
        break;
      }
    }
    if (!is_match) continue;
    matched_ops.push_back(op);
  }
  results.set(llvm::cast<mlir::OpResult>(getMatchedOps()), matched_ops);

  return mlir::DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FoldFillIntoPad
//===----------------------------------------------------------------------===//

// Ported from:
// iree/compiler/Codegen/Common/TransformExtensions/CommonExtensions.cpp
namespace {
/// Fold `tensor.pad(cst, tensor.extract*(linalg.fill(cst)))` into
/// `linalg.fill(cst, empty)` when the padding constant and the fill constant
/// are the same.
/// This seems generally desirable as a folding but may be too intrusive, so we
/// only apply it selectively for now.
// TODO: atm hardcoded on linalg.fill but we could take any result of any
// generic that yields a constant in that result.
struct FoldFillIntoPad : public mlir::OpRewritePattern<mlir::tensor::PadOp> {
  using mlir::OpRewritePattern<mlir::tensor::PadOp>::OpRewritePattern;
  mlir::LogicalResult matchAndRewrite(
      mlir::tensor::PadOp padOp, mlir::PatternRewriter &rewriter) const final {
    mlir::Operation *currentOp = padOp.getSource().getDefiningOp();
    auto maybeExtractSlice =
        mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(currentOp);
    while (currentOp && maybeExtractSlice) {
      currentOp = maybeExtractSlice.getSource().getDefiningOp();
      maybeExtractSlice =
          mlir::dyn_cast_or_null<mlir::tensor::ExtractSliceOp>(currentOp);
    }
    auto fillOp = mlir::dyn_cast_or_null<mlir::linalg::FillOp>(currentOp);
    if (!fillOp) {
      return rewriter.notifyMatchFailure(
          padOp, "not coming from a linalg.fill op via tensor.extract_slice*");
    }

    mlir::Value padValue = padOp.getConstantPaddingValue();
    mlir::RankedTensorType resultType = padOp.getResultType();
    if (!padValue ||
        getAsOpFoldResult(padValue) !=
            getAsOpFoldResult(fillOp.getDpsInputOperand(0)->get())) {
      return rewriter.notifyMatchFailure(
          padOp, "not a constant value matching the fill value");
    }

    mlir::Location loc = padOp.getLoc();
    auto emptyOp = rewriter.create<mlir::tensor::EmptyOp>(
        loc, mlir::tensor::getMixedSizes(rewriter, loc, padOp),
        resultType.getElementType());
    rewriter.replaceOpWithNewOp<mlir::linalg::FillOp>(padOp, padValue,
                                                      emptyOp.getResult());

    return mlir::success();
  }
};
}  // namespace

void jasc::ApplyFoldFillIntoPadPatternsOp::populatePatterns(
    mlir::RewritePatternSet &patterns) {
  patterns.insert<FoldFillIntoPad>(patterns.getContext());
}

//===----------------------------------------------------------------------===//
// SynchronizeOp
//===----------------------------------------------------------------------===//

void jasc::SynchronizeOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getOp(), effects);
  mlir::transform::producesHandle(getBarrier(), effects);
  mlir::transform::modifiesPayload(effects);
}

mlir::DiagnosedSilenceableFailure jasc::SynchronizeOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, mlir::Operation *operation,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  rewriter.setInsertionPointAfter(operation);
  auto barrier = rewriter.create<mlir::gpu::BarrierOp>(operation->getLoc());
  results.push_back(barrier);
  return mlir::DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// TuningParamOp
//===----------------------------------------------------------------------===//

mlir::DiagnosedSilenceableFailure jasc::TuningParamOp::apply(
    mlir::transform::TransformRewriter &rewriter,
    mlir::transform::TransformResults &results,
    mlir::transform::TransformState &state) {
  if (!getTunedValue().has_value()) {
    mlir::emitWarning(getLoc())
        << "tuning param not tuned, falling back to default value";
  }
  results.setParams(llvm::cast<mlir::OpResult>(getParam()),
                    {getTunedValue().value_or(getDefaultValue())});
  return mlir::DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ApplyTuningConfigOp
//===----------------------------------------------------------------------===//

mlir::DiagnosedSilenceableFailure jasc::ApplyTuningConfigOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, mlir::Operation *operation,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  size_t configIdx = 0;
  auto config = llvm::cast<mlir::ArrayAttr>(getConfig());
  // Walk all tuning parameters and set their value according to the config
  // attribute
  mlir::WalkResult walkResult = operation->walk<mlir::WalkOrder::PreOrder>(
      [&](jasc::TuningParamOp tuningParamOp) {
        if (configIdx >= config.size()) {
          return mlir::WalkResult::interrupt();
        }
        auto configVal = config[configIdx++];
        tuningParamOp.setTunedValueAttr(configVal);
        return mlir::WalkResult::skip();
      });
  if (configIdx == 0) {
    operation->emitWarning()
        << "no tuning parameters found, expected " << config.size();
    return mlir::DiagnosedSilenceableFailure::success();
  }
  if (walkResult.wasInterrupted() || configIdx != config.size()) {
    return mlir::emitSilenceableFailure(getLoc())
           << "size of config has to match the number of tunable variables: "
           << config.size() << " vs " << configIdx;
  }
  return mlir::DiagnosedSilenceableFailure::success();
}

void jasc::ApplyTuningConfigOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getTarget(), effects);
}
//===----------------------------------------------------------------------===//
// WrapInGpuLaunchOp
//===----------------------------------------------------------------------===//

mlir::DiagnosedSilenceableFailure jasc::WrapInGpuLaunchOp::applyToOne(
    mlir::transform::TransformRewriter &rewriter, mlir::Operation *operation,
    mlir::transform::ApplyToEachResultList &results,
    mlir::transform::TransformState &state) {
  mlir::Location loc = operation->getLoc();

  if (!operation->getUsers().empty()) {
    return mlir::emitSilenceableFailure(loc)
           << "The operation has users, cannot wrap the operation in a "
              "gpu.launch";
  }

  if (auto existingLaunchOp =
          operation->getParentOfType<mlir::gpu::LaunchOp>()) {
    mlir::DiagnosedSilenceableFailure diag =
        mlir::emitSilenceableFailure(loc)
        << "not wrapping this op into a gpu.launch op because it already is "
           "contained in one.";
    diag.attachNote(existingLaunchOp->getLoc())
        << "contained in this gpu.launch op.";
    return diag;
  }

  rewriter.setInsertionPoint(operation);
  auto one = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 1);
  auto launch_op =
      rewriter.create<mlir::gpu::LaunchOp>(loc, one, one, one, one, one, one);
  rewriter.setInsertionPointToEnd(&launch_op.getBody().front());
  auto terminator = rewriter.create<mlir::gpu::TerminatorOp>(loc);
  operation->moveBefore(terminator);

  results.push_back(launch_op);
  return mlir::DiagnosedSilenceableFailure::success();
}

void jasc::WrapInGpuLaunchOp::getEffects(
    llvm::SmallVectorImpl<mlir::MemoryEffects::EffectInstance> &effects) {
  mlir::transform::onlyReadsHandle(getOps(), effects);
  mlir::transform::producesHandle(getGpuLaunch(), effects);
}
