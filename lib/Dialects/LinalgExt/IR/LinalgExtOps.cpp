//===-- LinalgExtOps.h - Linalg Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgExt/LinalgExtOps.h"

#include "Dialects/LinalgExt/LinalgExtInterfaces.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"

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

SmallVector<Range> ReverseOp::getIterationDomain(OpBuilder &builder) {
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

SmallVector<Operation *> ReverseOp::getTiledImplementation(
    OpBuilder &builder, ValueRange outputs, ArrayRef<OpFoldResult> offsets,
    ArrayRef<OpFoldResult> sizes, bool tileDestOperands) {
  if (outputs.size() != 1) {
    this->emitOpError("expected single destination while tiling operation");
    return {};
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
  if (tileDestOperands)
    tiledOperands.emplace_back(
        getSlice(builder, loc, outputs[0], mirrorOffsets, sizes, strides));
  else
    tiledOperands.emplace_back(outputs[0]);

  if (hasTensorSemantics())
    resultTypes.push_back(tiledOperands.back().getType());

  Operation *tiledRevOp = cast<LinalgExtOp>(getOperation())
                              .clone(builder, loc, resultTypes, tiledOperands);

  if (tileDestOperands) {
    for (auto result : llvm::enumerate(tiledRevOp->getResults())) {
      builder.create<tensor::InsertSliceOp>(loc, result.value(),
                                            outputs[result.index()],
                                            mirrorOffsets, sizes, strides);
    }
  }
  return {tiledRevOp};
}

//===----------------------------------------------------------------------===//
// TileOp
//===----------------------------------------------------------------------===//

void TileOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                   Value tileSize, ValueRange outs, int64_t tiledDim,
                   TileOp::TileOpBodyBuilderFn bodyBuilder) {
  result.addOperands(tileSize);
  result.addOperands(outs);
  result.addAttribute(TileOp::getTiledDimAttrName(),
                      builder.getI64IntegerAttr(tiledDim));
  result.addTypes(outs.getType());

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  // TODO: Pass a better location here.
  Location loc = tileSize.getLoc();
  bodyBlock.addArgument(builder.getIndexType(), loc);
  bodyBlock.addArgument(builder.getIndexType(), loc);
  // Handle the sliced out types in a conservative fashion: all dimensions
  // become dynamic and a later canonicalization is expected to recover static
  // types.
  // TODO: should we relax this and use something less strict?
  auto dynamicTypes =
      llvm::to_vector(llvm::map_range(outs.getTypes(), [](Type t) -> Type {
        auto rankedTensorType = t.cast<RankedTensorType>();
        RankedTensorType::Builder rttb(rankedTensorType);
        SmallVector<int64_t> dynamicShape(rankedTensorType.getRank(),
                                          ShapedType::kDynamicSize);
        return rttb.setShape(dynamicShape);
      }));
  SmallVector<Location> locs(dynamicTypes.size(), loc);
  bodyBlock.addArguments(dynamicTypes, locs);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  bodyBuilder(builder, result.location, bodyBlock.getArgument(0),
              bodyBlock.getArgument(1), bodyBlock.getArguments().drop_front(2));
}

void TileOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                   Value tileSize, ValueRange outs,
                   TileOp::TileOpBodyBuilderFn bodyBuilder) {
  TileOp::build(builder, result, tileSize, outs, 0, bodyBuilder);
}

// TODO(#81): Impl me.
LogicalResult mlir::linalg_ext::TileOp::verify() { return success(); }

void mlir::linalg_ext::TileOp::print(OpAsmPrinter &p) {
  p << ' ' << tile_size() << ' ';
  if (tiled_dim() > 0)
    p << "tiled_dim = " << tiled_dim() << ' ';
  if (!outs().empty()) {
    p << "outs(";
    llvm::interleaveComma(outs(), p,
                          [&p](Value v) { p << v << ": " << v.getType(); });
    p << ')';
  }
  p << " -> (" << getResultTypes() << ") ";
  p.printRegion(region(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(getOperation()->getAttrs(),
                          /*elidedAttrs=*/{TileOp::getTiledDimAttrName()});
}

ParseResult mlir::linalg_ext::TileOp::parse(OpAsmParser &parser,
                                            OperationState &result) {
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

  // Parse the `tiled_dim` attribute or set it to 0 implicitly when elided.
  if (succeeded(parser.parseOptionalKeyword(TileOp::getTiledDimAttrName()))) {
    outputsOperandsLoc = parser.getCurrentLocation();
    Attribute valueAttr;
    parser.parseAttribute(valueAttr, TileOp::getTiledDimAttrName(),
                          result.attributes);
  } else {
    result.attributes.append(TileOp::getTiledDimAttrName(),
                             parser.getBuilder().getI64IntegerAttr(0));
  }

  if (succeeded(parser.parseOptionalKeyword("outs"))) {
    bool _1;
    SmallVector<NamedAttrList> _2;
    SmallVector<Location> _3;
    outputsOperandsLoc = parser.getCurrentLocation();
    if (mlir::function_interface_impl::parseFunctionArgumentList(
            parser,
            /*allowAttributes=*/false,
            /*allowVariadic=*/false, outsOperands, outsTypes, /*argAttrs=*/_2,
            /*argLocations=*/_3,
            /*isVariadic=*/_1) ||
        parser.resolveOperands(outsOperands, outsTypes, outputsOperandsLoc,
                               result.operands))
      return failure();
  }
  if (parser.parseArrowTypeList(result.types))
    return failure();

  SmallVector<OpAsmParser::OperandType, 8> regionOperands;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  SmallVector<Type, 8> operandTypes, regionTypes;
  if (parser.parseRegion(*region, regionOperands, regionTypes))
    return failure();

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  TileOp::ensureTerminator(*region, builder, result.location);
  result.addRegion(std::move(region));

  return success();
}

//===----------------------------------------------------------------------===//
// InParallelOp
//===----------------------------------------------------------------------===//

LogicalResult mlir::linalg_ext::InParallelOp::verify() {
  // Check that the body defines as single block argument for the thread index.
  auto *body = getBody();
  if (body->getNumArguments() != 1)
    return emitOpError("body expects exactly one argument");
  if (!body->getArgument(0).getType().isIndex())
    return emitOpError(
        "expected body first argument to be an index argument for "
        "the thread index");

  // Verify consistency between the result types and the terminator.
  auto terminatorTypes = getTerminator().yieldedTypes();
  auto opResults = getResults();
  if (opResults.size() != terminatorTypes.size())
    return emitOpError("produces ")
           << opResults.size() << " results, but its terminator yields "
           << terminatorTypes.size() << " values";
  unsigned i = 0;
  for (auto e : llvm::zip(terminatorTypes, opResults)) {
    if (std::get<0>(e) != std::get<1>(e).getType())
      return emitOpError() << "type mismatch between " << i
                           << "th result of in_parallel (" << std::get<0>(e)
                           << ") and " << i << "th result yielded by its "
                           << "terminator (" << std::get<1>(e).getType() << ")";
    i++;
  }

  return success();
}

void mlir::linalg_ext::InParallelOp::print(OpAsmPrinter &p) {
  p << ' ' << num_threads() << ' ';
  p << " -> (" << getResultTypes() << ") ";
  p.printRegion(region(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/true);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult mlir::linalg_ext::InParallelOp::parse(OpAsmParser &parser,
                                                  OperationState &result) {
  auto &builder = parser.getBuilder();

  OpAsmParser::OperandType numThreads;
  Type indexType = builder.getIndexType();

  if (parser.parseOperand(numThreads) ||
      parser.resolveOperand(numThreads, indexType, result.operands))
    return failure();
  if (parser.parseArrowTypeList(result.types))
    return failure();

  SmallVector<OpAsmParser::OperandType, 8> regionOperands;
  SmallVector<Type, 8> regionTypes;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, regionOperands, regionTypes))
    return failure();
  InParallelOp::ensureTerminator(*region, builder, result.location);
  result.addRegion(std::move(region));

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

// Bodyless builder, result types must be specified.
void InParallelOp::build(mlir::OpBuilder &builder, mlir::OperationState &result,
                         TypeRange resultTypes, Value numThreads) {
  // TODO: Pass better location.
  Location loc = numThreads.getLoc();
  result.addOperands(numThreads);

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType(), loc);

  // Create the default terminator if the builder is not provided and if the
  // iteration arguments are not provided. Otherwise, leave this to the caller
  // because we don't know which values to return from the loop.
  InParallelOp::ensureTerminator(*bodyRegion, builder, result.location);
  result.addTypes(resultTypes);
}

// Builder that takes a bodyBuilder lambda, result types are inferred from
// the terminator.
void InParallelOp::build(
    mlir::OpBuilder &builder, mlir::OperationState &result, Value numThreads,
    function_ref<void(OpBuilder &, Location, Value)> bodyBuilder) {
  // TODO: Pass better location.
  Location loc = numThreads.getLoc();
  result.addOperands(numThreads);

  Region *bodyRegion = result.addRegion();
  bodyRegion->push_back(new Block);
  Block &bodyBlock = bodyRegion->front();
  bodyBlock.addArgument(builder.getIndexType(), loc);

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  bodyBuilder(builder, result.location, bodyBlock.getArgument(0));
  auto terminator =
      llvm::cast<PerformConcurrentlyOp>(bodyBlock.getTerminator());
  result.addTypes(terminator.yieldedTypes());
}

// The ensureTerminator method generated by SingleBlockImplicitTerminator is
// unaware of the fact that our terminator also needs a region to be well
// formed. We override it here to ensure that we do the right thing.
void InParallelOp::ensureTerminator(Region &region, Builder &builder,
                                    Location loc) {
  OpTrait::SingleBlockImplicitTerminator<PerformConcurrentlyOp>::Impl<
      InParallelOp>::ensureTerminator(region, builder, loc);
  auto terminator =
      llvm::dyn_cast<PerformConcurrentlyOp>(region.front().getTerminator());
  PerformConcurrentlyOp::ensureTerminator(terminator.getRegion(), builder, loc);
}

PerformConcurrentlyOp InParallelOp::getTerminator() {
  return cast<PerformConcurrentlyOp>(getBody()->getTerminator());
}

//===----------------------------------------------------------------------===//
// ParallelInsertSliceOp
//===----------------------------------------------------------------------===//

// Build a ParallelInsertSliceOp with mixed static and dynamic entries.
void ParallelInsertSliceOp::build(OpBuilder &b, OperationState &result,
                                  Value source, Value dest,
                                  ArrayRef<OpFoldResult> offsets,
                                  ArrayRef<OpFoldResult> sizes,
                                  ArrayRef<OpFoldResult> strides,
                                  ArrayRef<NamedAttribute> attrs) {
  SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  SmallVector<Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets,
                             ShapedType::kDynamicStrideOrOffset);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes,
                             ShapedType::kDynamicSize);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides,
                             ShapedType::kDynamicStrideOrOffset);
  build(b, result, {}, source, dest, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getI64ArrayAttr(staticOffsets),
        b.getI64ArrayAttr(staticSizes), b.getI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a ParallelInsertSliceOp with dynamic entries.
void ParallelInsertSliceOp::build(OpBuilder &b, OperationState &result,
                                  Value source, Value dest, ValueRange offsets,
                                  ValueRange sizes, ValueRange strides,
                                  ArrayRef<NamedAttribute> attrs) {
  SmallVector<OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [](Value v) -> OpFoldResult { return v; }));
  SmallVector<OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [](Value v) -> OpFoldResult { return v; }));
  build(b, result, source, dest, offsetValues, sizeValues, strideValues);
}

namespace {
/// Pattern to rewrite a parallel_insert_slice op with constant arguments.
class ParallelInsertSliceOpConstantArgumentFolder final
    : public OpRewritePattern<ParallelInsertSliceOp> {
public:
  using OpRewritePattern<ParallelInsertSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ParallelInsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    // No constant operand, just return.
    if (llvm::none_of(insertSliceOp.getOperands(), [](Value operand) {
          return matchPattern(operand, matchConstantIndex());
        }))
      return failure();

    // At least one of offsets/sizes/strides is a new constant.
    // Form the new list of operands and constant attributes from the
    // existing.
    SmallVector<OpFoldResult> mixedOffsets(insertSliceOp.getMixedOffsets());
    SmallVector<OpFoldResult> mixedSizes(insertSliceOp.getMixedSizes());
    SmallVector<OpFoldResult> mixedStrides(insertSliceOp.getMixedStrides());
    canonicalizeSubViewPart(mixedOffsets, ShapedType::isDynamicStrideOrOffset);
    canonicalizeSubViewPart(mixedSizes, ShapedType::isDynamic);
    canonicalizeSubViewPart(mixedStrides, ShapedType::isDynamicStrideOrOffset);

    // Create the new op in canonical form.
    rewriter.replaceOpWithNewOp<ParallelInsertSliceOp>(
        insertSliceOp, insertSliceOp.source(), insertSliceOp.dest(),
        mixedOffsets, mixedSizes, mixedStrides);
    return success();
  }
};
} // namespace

void ParallelInsertSliceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.add<ParallelInsertSliceOpConstantArgumentFolder>(context);
}

//===----------------------------------------------------------------------===//
// PerformConcurrentlyOp
//===----------------------------------------------------------------------===//

// TODO(ntv,apaszke): Implement this
LogicalResult mlir::linalg_ext::PerformConcurrentlyOp::verify() {
  return success();
}

void mlir::linalg_ext::PerformConcurrentlyOp::print(OpAsmPrinter &p) {
  p << " ";
  p.printRegion(region(),
                /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/false);
  p.printOptionalAttrDict(getOperation()->getAttrs());
}

ParseResult
mlir::linalg_ext::PerformConcurrentlyOp::parse(OpAsmParser &parser,
                                               OperationState &result) {
  auto &builder = parser.getBuilder();

  SmallVector<OpAsmParser::OperandType, 8> regionOperands;
  SmallVector<Type, 8> regionTypes;
  std::unique_ptr<Region> region = std::make_unique<Region>();
  if (parser.parseRegion(*region, regionOperands, regionTypes))
    return failure();
  PerformConcurrentlyOp::ensureTerminator(*region, builder, result.location);
  result.addRegion(std::move(region));

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

SmallVector<Type> PerformConcurrentlyOp::yieldedTypes() {
  return llvm::to_vector(
      llvm::map_range(this->yieldingOps(), [](ParallelInsertSliceOp op) {
        return op.yieldedType();
      }));
}

SmallVector<ParallelInsertSliceOp> PerformConcurrentlyOp::yieldingOps() {
  SmallVector<ParallelInsertSliceOp> ret;
  for (Operation &op : *getBody()) {
    // TODO: interface when this grows up.
    if (auto sliceOp = llvm::dyn_cast<ParallelInsertSliceOp>(op)) {
      ret.push_back(sliceOp);
      continue;
    }
    if (auto endPerformOp = llvm::dyn_cast<EndPerformConcurrentlyOp>(op)) {
      continue;
    }
    llvm_unreachable("Unexpected operation in perform_concurrently");
  }
  return ret;
}

#define GET_OP_CLASSES
#include "Dialects/LinalgExt/LinalgExtOps.cpp.inc"
