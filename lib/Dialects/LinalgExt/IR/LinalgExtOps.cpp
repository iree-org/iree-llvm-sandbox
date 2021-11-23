//===-- LinalgExtOps.h - Linalg Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgExt/LinalgExtOps.h"

#include "Dialects/LinalgExt/LinalgExtInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::linalg_ext;

//===----------------------------------------------------------------------===//
// Utils.
//===----------------------------------------------------------------------===//

/// Returns a memref.subview or a tensor.extract_slice based on the type of the
/// `source`.
static Value getSlice(OpBuilder &b, Location loc, Value source,
                      ArrayRef<OpFoldResult> offsets,
                      ArrayRef<OpFoldResult> sizes,
                      ArrayRef<OpFoldResult> strides) {
  return TypeSwitch<Type, Value>(source.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return b.create<tensor::ExtractSliceOp>(loc, source, offsets, sizes,
                                                strides);
      })
      .Case<MemRefType>([&](MemRefType type) -> Value {
        return b.create<memref::SubViewOp>(loc, source, offsets, sizes,
                                           strides);
      })
      .Default([&](Type t) { return nullptr; });
}

Value getDimValue(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  return TypeSwitch<Type, Value>(v.getType())
      .Case<RankedTensorType>([&](RankedTensorType t) -> Value {
        return builder.create<tensor::DimOp>(loc, v, dim);
      })
      .Case<MemRefType>([&](MemRefType t) -> Value {
        return builder.create<memref::DimOp>(loc, v, dim);
      })
      .Default([&](Type t) { return Value(); });
}

OpFoldResult getDim(OpBuilder &builder, Location loc, Value v, int64_t dim) {
  auto t = v.getType().cast<ShapedType>();
  if (t.isDynamicDim(dim)) {
    return getDimValue(builder, loc, v, dim);
  }
  return builder.getI64IntegerAttr(t.getDimSize(dim));
}

//===----------------------------------------------------------------------===//
// ReverseOp
//===----------------------------------------------------------------------===//

SmallVector<StringRef> ReverseOp::getLoopIteratorTypes() {
  SmallVector<StringRef> iteratorTypes(getOperandRank(),
                                       getParallelIteratorTypeName());
  return iteratorTypes;
}

SmallVector<Range> ReverseOp::getLoopBounds(OpBuilder &builder) {
  Location loc = getLoc();
  Value zero = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Range> ranges;
  for (auto dim : llvm::seq<int64_t>(0, getOperandRank())) {
    Value ub = getDimValue(builder, loc, input(), dim);
    ranges.emplace_back(Range{zero, ub, one});
  }
  return ranges;
}

Operation *ReverseOp::getTiledImplementation(OpBuilder &builder,
                                             ValueRange outputs,
                                             ArrayRef<OpFoldResult> offsets,
                                             ArrayRef<OpFoldResult> sizes) {
  if (outputs.size() != 1) {
    this->emitOpError("expected single destination while tiling operation");
    return nullptr;
  }
  int64_t rank = getOperandRank();
  SmallVector<OpFoldResult> strides(rank, builder.getI64IntegerAttr(1));
  Location loc = getLoc();
  SmallVector<Value> tiledOperands;
  tiledOperands.emplace_back(
      getSlice(builder, loc, input(), offsets, sizes, strides));

  AffineExpr sym0, sym1, sym2;
  bindSymbols(builder.getContext(), sym0, sym1, sym2);
  AffineMap map =
      AffineMap::get(/*dimCount=*/0, /*symbolCount=*/3, {sym0 - sym1 - sym2});
  SmallVector<OpFoldResult> mirrorOffsets(offsets.begin(), offsets.end());
  for (auto dim : dims()) {
    Value size = getDimValue(builder, loc, input(), dim);
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, mirrorOffsets[dim]);
    Value tileSize = getValueOrCreateConstantIndexOp(builder, loc, sizes[dim]);
    mirrorOffsets[dim] =
        builder
            .create<AffineApplyOp>(loc, map, ValueRange{size, offset, tileSize})
            .getResult();
  }

  SmallVector<Type, 4> resultTypes;
  if (hasTensorSemantics()) {
    tiledOperands.emplace_back(
        getSlice(builder, loc, outputs[0], mirrorOffsets, sizes, strides));
    resultTypes.push_back(tiledOperands[1].getType());
  } else {
    tiledOperands.emplace_back(
        getSlice(builder, loc, outputs[0], mirrorOffsets, sizes, strides));
  }

  Operation *tiledRevOp = cast<LinalgExtOp>(getOperation())
                              .clone(builder, loc, resultTypes, tiledOperands);

  for (auto result : llvm::enumerate(tiledRevOp->getResults())) {
    builder.create<tensor::InsertSliceOp>(loc, result.value(),
                                          outputs[result.index()],
                                          mirrorOffsets, sizes, strides);
  }
  return tiledRevOp;
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

using BodyBuilderFn =
    function_ref<void(OpBuilder &, Location, Value /*offset*/, Value /*size*/)>;

static LogicalResult verify(TileOp op) { return success(); }

static void print(OpAsmPrinter &p, TileOp op) {
  p << ' ' << op.tile_sizes() << ' ';
  if (!op.outs().empty()) {
    p << "outs(";
    llvm::interleaveComma(op.outs(), p,
                          [&p](Value v) { p << v << ": " << v.getType(); });
    p << ')';
  }
  p << " -> (" << op.getResultTypes() << ')';
  p.printRegion(op.region(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseTileOp(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::OperandType tileSizes;
  // TODO: also allow tensor<..xindex> and figure out a good syntax.
  // Type tensorOfIndexType =
  //     RankedTensorType::get({ShapedType::kDynamicSize}, indexType);
  Type tileSizesType = builder.getIndexType();
  SmallVector<Type> outsTypes;
  SmallVector<OpAsmParser::OperandType, 4> outsOperands;

  llvm::SMLoc outputsOperandsLoc;
  if (parser.parseOperand(tileSizes) ||
      parser.resolveOperand(tileSizes, tileSizesType, result.operands))
    return failure();

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    bool _1;
    SmallVector<NamedAttrList> _2;
    outputsOperandsLoc = parser.getCurrentLocation();
    if (mlir::function_like_impl::parseFunctionArgumentList(
            parser,
            /*allowAttributes=*/false, /*allowVariadic=*/false, outsOperands,
            outsTypes, /*argAttrs=*/_2, /*isVariadic=*/_1) ||
        parser.resolveOperands(outsOperands, outsTypes, outputsOperandsLoc,
                               result.operands))
      return failure();
  }
  if (parser.parseArrowTypeList(result.types)) return failure();

  SmallVector<OpAsmParser::OperandType, 8> regionOperands;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  SmallVector<Type, 8> operandTypes, regionTypes;
  if (parser.parseRegion(*region, regionOperands, regionTypes))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes)) return failure();

  TileOp::ensureTerminator(*region, builder, result.location);
  result.addRegion(std::move(region));

  return success();
}

#define GET_OP_CLASSES
#include "Dialects/LinalgExt/LinalgExtOps.cpp.inc"
