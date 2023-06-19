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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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

struct AddPtrOpConversion : OpConversionPattern<triton::AddPtrOp> {
  AddPtrOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    Type idx = rewriter.getIndexType();

    // If the pointer got converted to an LLVM pointer, it's a scalar pointer.
    Type convertedPtrType = adaptor.getPtr().getType();
    if (convertedPtrType.isa<LLVMPointerType>()) {
      // Replace original op with LLVM's GEP op.
      Value basePtr = adaptor.getPtr();
      Value offset = adaptor.getOffset();
      rewriter.replaceOpWithNewOp<LLVM::GEPOp>(op, convertedPtrType, basePtr,
                                               offset);
      return success();
    }

    // Tensor of pointers.
    Type originalPtrType = op.getPtr().getType();
    if (auto ptrTensorType = originalPtrType.dyn_cast<RankedTensorType>()) {
      if (!ptrTensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            loc, "only static shapes supported for now");

      auto elementPtrType =
          ptrTensorType.getElementType().cast<triton::PointerType>();
      Type llvmPtrType = typeConverter->convertType(elementPtrType);
      Type llvmElementType =
          llvmPtrType.cast<LLVMPointerType>().getElementType();
      auto idxTensorType = adaptor.getPtr().getType().cast<RankedTensorType>();

      assert((llvmElementType.isIntOrFloat() ||
              llvmElementType.isa<LLVMPointerType>()) &&
             "expected int, float, or pointer as pointee types");

      // Compute element size in bytes.
      DataLayout dataLayout = DataLayout::closest(op);
      uint32_t elementTypeSize = dataLayout.getTypeSize(llvmElementType);
      Value offsets = rewriter.create<arith::IndexCastOp>(loc, idxTensorType,
                                                          adaptor.getOffset());
      Value elementSize =
          rewriter.create<arith::ConstantIndexOp>(loc, elementTypeSize);

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

    return rewriter.notifyMatchFailure(loc, "unsupported type of pointer");
  }
};

struct BitcastOpConversion : public OpConversionPattern<triton::BitcastOp> {
  BitcastOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto targetTensorType = op.getResult().getType().cast<TensorType>();
    Value input = adaptor.getFrom();

    assert(targetTensorType.getElementType().isIntOrFloat() &&
           "unexpected element type");
    assert(input.getType().cast<TensorType>().getElementType().isIntOrFloat() &&
           "unexpected element type");

    rewriter.replaceOpWithNewOp<tensor::BitcastOp>(op, targetTensorType, input);

    return success();
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
      // Collapse all source dims into the current result dim until (1) there
      // are no dims left and (2) either (a) we find one that shouldn't be
      // collapsed or (b) this is the last result dim, in which case the next
      // input dim must be one that should be collapsed.
      int64_t dimToPush;
      do {
        dimToPush = j++;
        reassociationMap[i].push_back(rewriter.getAffineDimExpr(dimToPush));
      } while ((j < rank) && (inputShape[dimToPush] != resultShape[dimToPush] ||
                              i == collapsedRank - 1));
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

struct DotOpConversion : public OpConversionPattern<triton::DotOp> {
  DotOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                  PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value inputA = op.getA();
    Value inputB = op.getB();
    Value inputC = op.getC();

    // Compute A * B + C with `linalg.matmul`.
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, /*ins=*/ValueRange{inputA, inputB}, /*outs=*/inputC);

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

struct LoadOpConversion : OpConversionPattern<triton::LoadOp> {
  LoadOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                   PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // If the pointer got converted to an LLVM pointer, it's a scalar pointer.
    Type convertedPtrType = adaptor.getPtr().getType();
    if (auto llvmPtrType = convertedPtrType.dyn_cast<LLVMPointerType>()) {
      // Unmasked load.
      if (!op.getMask()) {
        rewriter.replaceOpWithNewOp<LLVM::LoadOp>(op, adaptor.getPtr());
        return success();
      }

      // Masked load.
      rewriter.replaceOpWithNewOp<scf::IfOp>(
          op, /*condition=*/op.getMask(),
          /*thenBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            Value loaded = rewriter.create<LLVM::LoadOp>(loc, adaptor.getPtr());
            builder.create<scf::YieldOp>(loc, loaded);
          },
          /*elseBuilder=*/
          [&](OpBuilder &builder, Location loc) {
            if (op.getOther()) {
              builder.create<scf::YieldOp>(loc, op.getOther());
              return;
            }

            Type elemType = llvmPtrType.getElementType();
            Value undef = rewriter.create<LLVM::UndefOp>(loc, elemType);
            builder.create<scf::YieldOp>(loc, undef);
          });
      return success();
    }

    // Only handle unmasked pointers for now.
    if (op.getMask() || op.getOther())
      return rewriter.notifyMatchFailure(loc, "mask+other not supported yet");

    // Tensor of pointers.
    // TODO(ingomueller): This is a manual tiling by one. That is fine in order
    //     to get things running but drops a lot of information. Eventually, we
    //     want to map this to a vectorized load/gather in order to distribute
    //     the loading over SIMT threads.
    Type originalPointerType = op.getPtr().getType();
    if (auto tensorType = originalPointerType.dyn_cast<RankedTensorType>()) {
      if (!tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            loc, "only static shapes supported for now");

      // Derive types.
      auto elementPtrType =
          tensorType.getElementType().cast<triton::PointerType>();
      Type llvmPtrType = typeConverter->convertType(elementPtrType);
      Type resultType = typeConverter->convertType(op.getResult().getType());
      auto resultTensorType = resultType.cast<TensorType>();
      Type resultElementType = resultTensorType.getElementType();
      Type llvmElementType =
          llvmPtrType.cast<LLVMPointerType>().getElementType();

      assert((llvmElementType.isIntOrFloat() ||
              llvmElementType.isa<LLVMPointerType>()) &&
             "expected int, float, or pointer as pointee types");

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
          loc, tensorType.getShape(), resultElementType);
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

            // Convert element back to index if necessary. This happens on
            // pointer-of-pointer inputs.
            if (llvmElementType != resultElementType) {
              element = b.create<LLVM::PtrToIntOp>(loc, i64, element);
              element = b.create<arith::IndexCastOp>(loc, idx, element);
            }

            // Insert extracted value into result tensor.
            values = b.create<tensor::InsertOp>(loc, element, values, ivs);
            return SmallVector<Value>{values};
          });
      values = forOp.results[0];

      rewriter.replaceOp(op, values);
      return success();
    }

    return rewriter.notifyMatchFailure(loc, "unsupported type of pointer");
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

struct ReduceReturnOpConversion
    : public OpConversionPattern<triton::ReduceReturnOp> {
  ReduceReturnOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                           PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ReduceReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::YieldOp>(op, adaptor.getOperands());
    return success();
  }
};

struct ReduceOpConversion : public OpConversionPattern<triton::ReduceOp> {
  ReduceOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                     PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Derive types. `tt.reduce` treats reducing a 1-D tensor with a special
    // case that returns a scalar, but we treat it as a 0-D tensor in these
    // types.
    auto convertedInputTensorTypes =
        llvm::map_range(adaptor.getOperands().getTypes(),
                        [](Type t) { return t.cast<TensorType>(); });
    auto originalResultTensorTypes =
        llvm::map_range(op.getResultTypes(), [](Type t) -> TensorType {
          if (auto tensorType = t.dyn_cast<TensorType>())
            return tensorType;
          return RankedTensorType::get({}, t);
        });
    assert(llvm::all_equal(llvm::map_range(
        convertedInputTensorTypes, [](TensorType t) { return t.getShape(); })));
    assert(llvm::all_equal(llvm::map_range(
        originalResultTensorTypes, [](TensorType t) { return t.getShape(); })));
    ArrayRef<int64_t> resultShape =
        (*originalResultTensorTypes.begin()).getShape();
    auto convertedResultTensorTypes =
        llvm::map_range(originalResultTensorTypes, [&](TensorType t) {
          return RankedTensorType::get(resultShape, t.getElementType());
        });

    // Create empty vectors as init values.
    llvm::SmallVector<Value> initVals;
    for (TensorType t : convertedResultTensorTypes) {
      TypedAttr zeroAttr = rewriter.getZeroAttr(t.getElementType());
      Value zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
      auto emptyOp = rewriter.create<tensor::SplatOp>(loc, t, zero);
      initVals.push_back(emptyOp);
    }

    // Create a linalg.reduce on the same input and move the combine region
    // there. (ReduceReturnOpConversion will take care of the terminator.)
    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, /*resultTypes=*/SmallVector<Type>(convertedResultTensorTypes),
        /*inputs=*/adaptor.getOperands(), /*inits=*/initVals,
        /*dimensions=*/ArrayRef<int64_t>{op.getAxis()});
    rewriter.inlineRegionBefore(op.getCombineOp(), reduceOp.getCombiner(),
                                reduceOp.getCombiner().end());

    // If the result on tt.reduce are tensors with rank > 0, we are done.
    if (!resultShape.empty()) {
      rewriter.replaceOp(op, reduceOp);
      return success();
    }

    // Otherwise, the result has to be a scalar, so we need to extract the
    // scalar from the 0-ranked result tensor.
    SmallVector<Value> results;
    for (auto [tensor, type] :
         llvm::zip(reduceOp->getResults(), convertedResultTensorTypes)) {
      Value scalar = rewriter.create<tensor::ExtractOp>(
          loc, type.getElementType(), tensor, /*indices=*/ValueRange{});
      results.push_back(scalar);
    }
    rewriter.replaceOp(op, results);

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

struct StoreOpConversion : OpConversionPattern<triton::StoreOp> {
  StoreOpConversion(TypeConverter &typeConverter, MLIRContext *context,
                    PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult
  matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();

    // Only handle unmasked pointers for now.
    if (op.getMask())
      return rewriter.notifyMatchFailure(loc, "mask not supported yet");

    // If the pointer got converted to an LLVM pointer, it's a scalar pointer.
    Type convertedPtrType = adaptor.getPtr().getType();
    if (convertedPtrType.isa<LLVMPointerType>()) {
      rewriter.replaceOpWithNewOp<LLVM::StoreOp>(op, adaptor.getValue(),
                                                 adaptor.getPtr());
      return success();
    }

    // Tensor of pointers.
    // TODO(ingomueller): This is a manual tiling by one. That is fine in order
    //     to get things running but drops a lot of information. Eventually, we
    //     want to map this to a vectorized store/scatter in order to distribute
    //     the storing over SIMT threads.
    Type originalPointerType = op.getPtr().getType();
    if (auto tensorType = originalPointerType.dyn_cast<RankedTensorType>()) {
      if (!tensorType.hasStaticShape())
        return rewriter.notifyMatchFailure(
            loc, "only static shapes supported for now");

      // Derive types.
      auto elementPtrType =
          tensorType.getElementType().cast<triton::PointerType>();
      Type llvmPtrType = typeConverter->convertType(elementPtrType);
      Type llvmElementType =
          llvmPtrType.cast<LLVMPointerType>().getElementType();

      assert((llvmElementType.isIntOrFloat() ||
              llvmElementType.isa<LLVMPointerType>()) &&
             "expected int, float, or pointer as pointee types");

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

      // Store each tensor element at a time.
      scf::buildLoopNest(
          rewriter, loc, lbs, ubs, steps, ValueRange{},
          [&](OpBuilder &b, Location loc, ValueRange ivs, ValueRange args) {
            Type idx = b.getIndexType();
            Type i64 = b.getI64Type();

            // Extract value that should be stored.
            Value element =
                b.create<tensor::ExtractOp>(loc, adaptor.getValue(), ivs);

            //  Convert if necessary. This happens with pointers of pointers.
            if (element.getType().isIndex()) {
              element = b.create<arith::IndexCastOp>(loc, i64, element);
              element = b.create<LLVM::IntToPtrOp>(loc, llvmPtrType, element);
            }

            // Extract address, cast to pointer, and store value there.
            Value address =
                b.create<tensor::ExtractOp>(loc, idx, adaptor.getPtr(), ivs);
            address = b.create<arith::IndexCastOp>(loc, i64, address);
            address = b.create<LLVM::IntToPtrOp>(loc, llvmPtrType, address);
            rewriter.create<LLVM::StoreOp>(loc, element, address);

            return SmallVector<Value>();
          });
      rewriter.eraseOp(op);

      return success();
    }

    return rewriter.notifyMatchFailure(loc, "unsupported type of pointer");
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
      LoadOpConversion,
      StoreOpConversion
      // clang-format on
      >(typeConverter, patterns.getContext());

  patterns.add<
      // clang-format off
      BroadcastOpConversion,
      BitcastOpConversion,
      DotOpConversion,
      ExpandDimsOpConversion,
      MakeRangeOpConversion,
      ReduceOpConversion,
      ReduceReturnOpConversion,
      SplatOpConversion,
      ViewOpConversion
      // clang-format on
      >(typeConverter, patterns.getContext());
}

void ConvertTritonToLLVMPass::runOnOperation() {
  auto module = getOperation();

  // We have two type converters: the `TypeConverter` instance only convert
  // pointer types -- including the pointee types. If that same type, however,
  // is used outside an LLVM pointer, it should be handled differently. We thus
  // pass around the `TypeConverter` in the pass and, to implement its
  // conversions, use the `LLVMTypeConverter` for the pointee types.
  TypeConverter typeConverter;
  LLVMTypeConverter llvmTypeConverter(&getContext());

  // Leave unrelated types unchanged.
  typeConverter.addConversion([&](Type type) { return type; });

  // Scalar `triton::PointerType`.
  // TODO(ingomueller): We preserve the address space attribute here but we'll
  //     probably ignore its value in the conversions that use these pointers,
  //     so we'll have to revisit the whole concept of address spaces at some
  //     point.
  auto convertTritonPtrType =
      [&](triton::PointerType type) -> std::optional<Type> {
    Type pointeeType = type.getPointeeType();
    Type convertedPointeeType = llvmTypeConverter.convertType(pointeeType);

    if (!LLVMPointerType::isValidElementType(convertedPointeeType))
      return std::nullopt;

    return LLVMPointerType::get(convertedPointeeType, type.getAddressSpace());
  };
  typeConverter.addConversion(convertTritonPtrType);
  llvmTypeConverter.addConversion(convertTritonPtrType);

  // Tensor of `triton::PointerType`s.
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

  // Add patterns that convert the types in SCF constructs.
  scf::populateSCFStructuralTypeConversionsAndLegality(typeConverter, patterns,
                                                       target);

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
