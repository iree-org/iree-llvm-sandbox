//===-- TritonToLLVM.cpp - Conversion from Triton to LLVM -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Conversion/TritonToLLVM/TritonToLLVM.h"

#include "../PassDetail.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include <numeric>

namespace mlir {
class MLIRContext;
} // namespace mlir

using namespace mlir;
using namespace mlir::arith;
using namespace mlir::linalg;
using namespace mlir::LLVM;
using namespace mlir::scf;
using namespace mlir::tensor;
using namespace triton;

namespace {
struct ConvertTritonToLLVMPass
    : public ConvertTritonToLLVMBase<ConvertTritonToLLVMPass> {
  void runOnOperation() override;
};

struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  AddPtrOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type idx = rewriter.getIndexType();
    Type ptrType = op.getPtr().getType();

    // Scalar pointer.
    if (auto ttPtrType = ptrType.dyn_cast<triton::PointerType>()) {
      assert(ttPtrType.getPointeeType().isa<IntegerType>() &&
             "expected tt.ptr to point to an integer type");
      // Replace original op with LLVM's GEP op.
      Value basePtr = adaptor.getPtr();
      Value offset = adaptor.getOffset();
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, basePtr.getType(), basePtr,
                                               offset);
      return success();
    }

    // Tensor of pointers.
    if (auto ptrTensorType = ptrType.dyn_cast<RankedTensorType>()) {
      if (!ptrTensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            loc, "only static shapes supported for now");

      auto elementPtrType =
          ptrTensorType.getElementType().cast<triton::PointerType>();
      auto elementType = elementPtrType.getPointeeType();
      auto idxTensorType = adaptor.getPtr().getType().cast<RankedTensorType>();

      // Compute element size in bytes.
      uint32_t elementBitWidth = elementType.getIntOrFloatBitWidth();
      Value offsets = rewriter.create<arith::IndexCastOp>(loc, idxTensorType,
                                                          adaptor.getOffset());
      Value elementSize =
          rewriter.create<arith::ConstantIndexOp>(loc, elementBitWidth / 8);

      // Compute offsets in terms of bytes.
      Type offsetsTensorType =
          RankedTensorType::get(idxTensorType.getShape(), idx);
      elementSize =
          rewriter.create<tensor::SplatOp>(loc, elementSize, offsetsTensorType);
      offsets = rewriter.create<arith::MulIOp>(loc, offsets, elementSize);

      // Add offsets to base pointer.
      rewriter.replaceOpWithNewOp<arith::AddIOp>(op, adaptor.getPtr(), offsets);

      return success();
    }

    return failure();
  }
};

struct BroadcastOpConversion : public OpConversionPattern<triton::BroadcastOp> {
  BroadcastOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Compute shapes and element type.
    auto inputShape = op.getSrc().getType().cast<ShapedType>().getShape();
    auto resultShape = op.getResult().getType().cast<ShapedType>().getShape();
    int32_t rank = static_cast<int32_t>(inputShape.size());
    assert(inputShape.size() == resultShape.size() &&
           "expected input and result to be of the same rank");
    auto elementType =
        adaptor.getSrc().getType().cast<ShapedType>().getElementType();

    // Find broadcast dimensions: dimensions that are different between the
    // input and the result (and 1 on the input).
    SmallVector<int64_t> broadcastDims;
    for (auto [idx, dims] :
         llvm::enumerate(llvm::zip(inputShape, resultShape))) {
      auto [inputDim, resultDim] = dims;
      if (inputDim != resultDim) {
        assert(inputDim == 1 &&
               "expected all differing dimensions to be broadcast dimensions");
        broadcastDims.push_back(idx);
      }
    }

    // `tensor.broadcast` uses non-existing dimensions whereas Triton uses
    // dimensions with unit extent, so we need to collapse those first before
    // being able to use `tensor.broadcast`.
    int32_t collapsedRank = rank - broadcastDims.size();
    SmallVector<ReassociationExprs> reassociationMap(collapsedRank);
    for (int64_t i = 0, j = 0; i < collapsedRank && j < rank; i++) {
      // Collapse all source dims into the current result dim until (1) we find
      // one that shouldn't be collapsed or (2) there are no dims left.
      int64_t dimToPush;
      do {
        dimToPush = j++;
        reassociationMap[i].push_back(rewriter.getAffineDimExpr(dimToPush));
      } while (inputShape[dimToPush] != resultShape[dimToPush] &&
               dimToPush < rank);
    }
    auto collapseOp = rewriter.create<tensor::CollapseShapeOp>(
        loc, adaptor.getSrc(), reassociationMap);

    // Broadcast collapsed value to empty tensor of desired shape.
    auto init = rewriter.create<tensor::EmptyOp>(loc, resultShape, elementType);
    rewriter.replaceOpWithNewOp<linalg::BroadcastOp>(op, collapseOp, init,
                                                     broadcastDims);

    return success();
  }
};

struct ExpandDimsOpConversion
    : public OpConversionPattern<triton::ExpandDimsOp> {
  ExpandDimsOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                         PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Compute types and shapes.
    auto originalTensorType =
        adaptor.getSrc().getType().cast<RankedTensorType>();
    ArrayRef<int64_t> expandedShape =
        op.getResult().getType().cast<ShapedType>().getShape();
    Type elementType = originalTensorType.getElementType();
    auto expandedTensorType = RankedTensorType::get(expandedShape, elementType);

    // Create association map: expand the desired axis to [axis, axis+1] and
    // keep the remaining ones as is/shifted.
    int64_t originalRank = originalTensorType.getRank();
    auto dimToExpand = std::min<int64_t>(op.getAxis(), originalRank - 1);
    SmallVector<ReassociationExprs> reassociationMap(originalRank);
    for (int64_t i = 0, j = 0; i < originalRank; i++, j++) {
      reassociationMap[i].push_back(rewriter.getAffineDimExpr(j));
      if (dimToExpand == i)
        reassociationMap[i].push_back(rewriter.getAffineDimExpr(++j));
    }

    // Replace with expand_shape.
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        op, expandedTensorType, adaptor.getSrc(), reassociationMap);

    return success();
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  LoadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle unmasked pointers for now.
    if (op.getMask() || op.getOther())
      return failure();

    Location loc = op->getLoc();
    Type ptrType = op.getPtr().getType();

    // Scalar pointer.
    if (auto ttPtrType = ptrType.dyn_cast<triton::PointerType>()) {
      if (ttPtrType.getPointeeType().isIntOrIndexOrFloat()) {
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, adaptor.getPtr());
        return success();
      }
    }

    // Tensor of pointers.
    // TODO(ingomueller): This is a manual tiling by one. That is fine in order
    //     to get things running but drops a lot of information. Eventually, we
    //     want to map this to a vectorized load/gather in order to distribute
    //     the loading over SIMT threads.
    if (auto tensorType = ptrType.dyn_cast<RankedTensorType>()) {
      if (!tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            loc, "only static shapes supported for now");

      // Derive types.
      Type elementType =
          op.getResult().getType().cast<TensorType>().getElementType();
      auto elementPtrType =
          tensorType.getElementType().cast<triton::PointerType>();
      auto llvmPtrType = typeConverter->convertType(elementPtrType);

      // Compute bounds of for loop.
      SmallVector<Value> steps(tensorType.getRank());
      SmallVector<Value> lbs(tensorType.getRank());
      SmallVector<Value> ubs(tensorType.getRank());
      for (auto &step : steps)
        step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      for (auto &lb : lbs)
        lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      for (int64_t i = 0; i < tensorType.getRank(); i++) {
        ubs[i] = rewriter.create<arith::ConstantIndexOp>(
            loc, tensorType.getDimSize(i));
      }

      // Load each tensor element at a time.
      Value values = rewriter.create<tensor::EmptyOp>(
          loc, tensorType.getShape(), elementType);
      LoopNest forOp = scf::buildLoopNest(
          rewriter, loc, lbs, ubs, steps, values,
          [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange args) {
            Value values = args[0];
            Type idx = b.getIndexType();
            Type i64 = b.getI64Type();

            // Extract index, convert to pointer, and load from there.
            Value address =
                b.create<tensor::ExtractOp>(loc, idx, adaptor.getPtr(), ivs);
            address = b.create<arith::IndexCastOp>(loc, i64, address);
            address = b.create<LLVM::IntToPtrOp>(loc, llvmPtrType, address);
            Value element = rewriter.create<LLVM::LoadOp>(loc, address);

            // Insert extracted value into result tensor.
            values = b.create<tensor::InsertOp>(loc, element, values, ivs);
            return SmallVector<Value>{values};
          });
      values = forOp.results[0];

      rewriter.replaceOp(op, values);
      return success();
    }

    return failure();
  }
};

struct MakeRangeOpConversion : public OpConversionPattern<triton::MakeRangeOp> {
  MakeRangeOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                        PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tensorType = op.getResult().getType().cast<TensorType>();

    // Create dense attribute with range values.
    int64_t num_elements = tensorType.getShape()[0];
    assert(num_elements == op.getEnd() - op.getStart());
    int32_t start = op.getStart();
    SmallVector<int32_t> values(num_elements);
    std::iota(values.begin(), values.end(), start);
    auto valuesAttr = DenseIntElementsAttr::get(tensorType, values);

    // Replace range op with new constant tensor.
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, tensorType, valuesAttr);

    return success();
  }
};

struct SplatOpConversion : public OpConversionPattern<triton::SplatOp> {
  SplatOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type i64 = rewriter.getI64Type();
    Type idx = rewriter.getIndexType();

    Value src = adaptor.getSrc();
    TensorType tensorType;

    // Depending on element type: conversion of splat value and tensor type.
    if (src.getType().isa<LLVMPointerType>()) {
      // Pointers.
      Type originalResultType = op.getResult().getType();
      Type convertedResultType = typeConverter->convertType(originalResultType);
      tensorType = convertedResultType.cast<TensorType>();
      assert(tensorType.getElementType() == idx);

      // Convert pointer to int, then cast to index.
      src = rewriter.create<LLVM::PtrToIntOp>(loc, i64, src);
      src = rewriter.create<arith::IndexCastOp>(loc, idx, src);
    } else {
      // Numeric scalars.
      assert(src.getType().isIntOrFloat());
      tensorType = op.getResult().getType().cast<TensorType>();
    }

    // Replace tt.splat with tensor.splat.
    rewriter.replaceOpWithNewOp<tensor::SplatOp>(op, src, tensorType);

    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  StoreOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only handle unmasked pointers for now.
    if (op.getMask())
      return failure();

    Location loc = op->getLoc();
    Type ptrType = op.getPtr().getType();

    // Scalar pointer.
    if (auto ttPtrType = ptrType.dyn_cast<triton::PointerType>()) {
      if (ttPtrType.getPointeeType().isIntOrIndexOrFloat()) {
        rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                                   adaptor.getPtr());
        return success();
      }
    }

    // Tensor of pointers.
    // TODO(ingomueller): This is a manual tiling by one. That is fine in order
    //     to get things running but drops a lot of information. Eventually, we
    //     want to map this to a vectorized store/scatter in order to distribute
    //     the storing over SIMT threads.
    if (auto tensorType = ptrType.dyn_cast<RankedTensorType>()) {
      if (!tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            loc, "only static shapes supported for now");
      if (tensorType.getRank() != 1)
        return rewriter.notifyMatchFailure(loc,
                                           "only 1D tensors supported for now");

      // Derive types.
      Type elementType =
          op.getValue().getType().cast<TensorType>().getElementType();
      auto elementPtrType =
          tensorType.getElementType().cast<triton::PointerType>();
      auto llvmPtrType = typeConverter->convertType(elementPtrType);

      // Compute bounds of for loop.
      Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ub = rewriter.create<arith::ConstantIndexOp>(
          loc, tensorType.getDimSize(0));
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

      // Store each tensor element at a time.
      rewriter.create<scf::ForOp>(
          loc, lb, ub, step, ValueRange{},
          [&](OpBuilder &b, Location loc, Value iv, ValueRange args) {
            Type idx = b.getIndexType();
            Type i64 = b.getI64Type();

            // Extract value that should be stored
            Value element = b.create<tensor::ExtractOp>(loc, elementType,
                                                        adaptor.getValue(), iv);

            // Extract address, cast to pointer, and store value there.
            Value address =
                b.create<tensor::ExtractOp>(loc, idx, adaptor.getPtr(), iv);
            address = b.create<arith::IndexCastOp>(loc, i64, address);
            address = b.create<LLVM::IntToPtrOp>(loc, llvmPtrType, address);
            rewriter.create<LLVM::StoreOp>(loc, element, address);

            b.create<scf::YieldOp>(loc);
          });
      rewriter.eraseOp(op);

      return success();
    }

    return failure();
  }
};

struct ViewOpConversion : public OpConversionPattern<triton::ViewOp> {
  ViewOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Compute converted result type.
    auto srcTensorType = adaptor.getSrc().getType().cast<RankedTensorType>();
    auto elementType = srcTensorType.getElementType();
    ArrayRef<int64_t> shape =
        op.getResult().getType().cast<ShapedType>().getShape();
    auto resultType = RankedTensorType::get(shape, elementType);

    // Create constant tensor describing new shape.
    DenseIntElementsAttr shapeAttr = rewriter.getIndexTensorAttr(shape);
    auto shapeTensor = rewriter.create<arith::ConstantOp>(loc, shapeAttr);

    // Replace op with reshape op.
    rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(
        op, resultType, adaptor.getSrc(), shapeTensor);

    return success();
  }
};
} // namespace

void mlir::populateTritonToLLVMConversionPatterns(
    RewritePatternSet &patterns, TypeConverter &typeConverter) {
  patterns.add<
      // clang-format off
      AddPtrOpConversion,
      BroadcastOpConversion,
      ExpandDimsOpConversion,
      LoadOpConversion,
      MakeRangeOpConversion,
      SplatOpConversion,
      StoreOpConversion,
      ViewOpConversion
      // clang-format on
      >(typeConverter, patterns.getContext());
}

void ConvertTritonToLLVMPass::runOnOperation() {
  auto module = getOperation();
  LLVMTypeConverter typeConverter(&getContext());

  // triton::PointerType: Replicate logic from
  // TritonGPUToLLVMTypeConverter::convertTritonPointerType.
  // TODO(ingomueller): We preserve the address space attribute here but we'll
  //     probably ignore its value in the conversions that use these pointers,
  //     so we'll have to revisit the whole concept of address spaces at some
  //     point.
  typeConverter.addConversion([&](triton::PointerType type) {
    return LLVM::LLVMPointerType::get(
        typeConverter.convertType(type.getPointeeType()),
        type.getAddressSpace());
  });
  // TODO(ingomueller): This drops the address space attribute. Is that a
  //     problem?
  // TODO(ingomueller): This converts a pointer to an index whose value is the
  //     address of the pointer. While this covers the general case, very often
  //     the pointers belong to a single allocation, which could be represented
  //     as a base pointer and a tensor of offsets. That, in turn, would
  //     preserve the semantics about the loads being local to each other and
  //     maybe fit to (to be developped) primitives in the indexing dialect.
  typeConverter.addConversion([&](RankedTensorType type) -> Type {
    if (auto ptrType = type.getElementType().dyn_cast<triton::PointerType>()) {
      auto idx = IndexType::get(type.getContext());
      return RankedTensorType::get(type.getShape(), idx);
    }
    return type;
  });

  // Convert the remaining ops of this dialect using dialect conversion.
  ConversionTarget target(getContext());
  target.addLegalDialect<ArithDialect, LinalgDialect, LLVMDialect, SCFDialect,
                         TensorDialect>();
  target.addLegalOp<ModuleOp>();
  RewritePatternSet patterns(&getContext());

  // Load patterns specific this pass.
  populateTritonToLLVMConversionPatterns(patterns, typeConverter);

  // Add patterns that converts function signature and calls.
  populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                 typeConverter);
  populateCallOpTypeConversionPattern(patterns, typeConverter);
  populateReturnOpTypeConversionPattern(patterns, typeConverter);

  // Force application of that pattern if signature is not legal yet.
  target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType());
  });
  target.addDynamicallyLegalOp<func::ReturnOp>([&](func::ReturnOp op) {
    return typeConverter.isLegal(op.getOperandTypes());
  });
  target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
    return typeConverter.isSignatureLegal(op.getCalleeType());
  });

  // Use UnrealizedConversionCast as materializations, which have to be cleaned
  // up by later passes.
  auto addUnrealizedCast = [](OpBuilder &builder, Type type, ValueRange inputs,
                              Location loc) {
    auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
    return std::optional<Value>(cast.getResult(0));
  };
  typeConverter.addSourceMaterialization(addUnrealizedCast);
  typeConverter.addTargetMaterialization(addUnrealizedCast);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::createConvertTritonToLLVMPass() {
  return std::make_unique<ConvertTritonToLLVMPass>();
}
