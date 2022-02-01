//===-- VectorExtOps.h - Vector Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/VectorExt/VectorExtOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::vector_ext;

//===----------------------------------------------------------------------===//
// PredicateOp
//===----------------------------------------------------------------------===//

/// Default callback for PredicateOp builders. Inserts a yield without
/// arguments.
void mlir::vector_ext::buildTerminatedBody(OpBuilder &builder, Location loc) {
  builder.create<vector_ext::YieldOp>(loc);
}

void PredicateOp::build(OpBuilder &builder, OperationState &result,
                        Value predicateMask, ValueRange indices,
                        Value incomingMask) {
  build(builder, result, /*resultTypes=*/llvm::None, predicateMask, indices,
        incomingMask);
}

void PredicateOp::build(
    OpBuilder &builder, OperationState &result, TypeRange resultTypes,
    Value predicateMask, ValueRange indices, Value incomingMask,
    function_ref<void(OpBuilder &, Location)> truePredicateBuilder) {
  assert(truePredicateBuilder &&
         "the builder callback for 'truePredicate' must be present");

  result.addOperands(predicateMask);
  result.addOperands(indices);
  result.addOperands(incomingMask);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *truePredicateRegion = result.addRegion();
  Block *bodyBlock = builder.createBlock(truePredicateRegion);
  bodyBlock->addArgument(predicateMask.getType(), result.location);
  truePredicateBuilder(builder, result.location);
}

static ParseResult parsePredicateOp(OpAsmParser &parser,
                                    OperationState &result) {
  // Create the regions for 'truePredicate'.
  result.regions.reserve(1);
  Region *truePredicateRegion = result.addRegion();

  auto &builder = parser.getBuilder();

  // Parse all the operands.
  OpAsmParser::OperandType predicateMask;
  OpAsmParser::OperandType incomingMask;
  SmallVector<OpAsmParser::OperandType> indices;
  if (parser.parseLParen() || parser.parseRegionArgument(predicateMask) ||
      parser.parseComma() ||
      parser.parseOperandList(indices, AsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseRegionArgument(incomingMask) ||
      parser.parseRParen())
    return failure();

  // Parse predicate type.
  Type maskType;
  if (parser.parseColonType(maskType))
    return failure();

  if (parser.resolveOperand(predicateMask, maskType, result.operands) ||
      parser.resolveOperands(indices, IndexType::get(builder.getContext()),
                             result.operands) ||
      parser.resolveOperand(incomingMask, maskType, result.operands))
    return failure();

  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();

  // Parse the 'truePredicate' region.
  if (parser.parseRegion(*truePredicateRegion, /*arguments=*/{},
                         /*argTypes=*/{}))
    return failure();
  PredicateOp::ensureTerminator(*truePredicateRegion, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

static void print(OpAsmPrinter &p, PredicateOp op) {
  bool printBlockTerminators = false;

  p << "(" << op.predicateMask() << ", [" << op.indices() << "], "
    << op.incomingMask() << ") : " << op.predicateMask().getType();
  if (!op.results().empty()) {
    p << " -> (" << op.getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << " ";
  p.printRegion(op.truePredicateRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/printBlockTerminators);

  p.printOptionalAttrDict(op->getAttrs());
}

/// Given the region at `index`, or the parent operation if `index` is None,
/// return the successor regions. These are the regions that may be selected
/// during the flow of control. `operands` is a set of optional attributes that
/// correspond to a constant value for each operand, or null if that operand is
/// not a constant.
void PredicateOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  // The `truePredicate` region branch back to the parent operation.
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // The `truePredicate` (and the future `falsePredicate` region)  will always
  // be executed regardless of the condition since they are not modeling control
  // but data flow.
  regions.push_back(RegionSuccessor(&truePredicateRegion()));
}

//===----------------------------------------------------------------------===//
// WarpSingleLaneOp
//===----------------------------------------------------------------------===//

static void print(OpAsmPrinter &p, WarpSingleLaneOp op) {
  p << "(" << op.laneid() << ")";
  if (!op.args().empty())
    p << " args(" << op.args() << " : " << op.args().getTypes() << ")";
  if (!op.results().empty())
    p << " -> (" << op.results().getTypes() << ')';
  p << " ";
  p.printRegion(op.getRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/!op.results().empty());
  p.printOptionalAttrDict(op->getAttrs());
}

static ParseResult parseWarpSingleLaneOp(OpAsmParser &parser,
                                         OperationState &result) {
  // Create the region.
  result.regions.reserve(1);
  Region *warpRegion = result.addRegion();

  auto &builder = parser.getBuilder();
  OpAsmParser::OperandType laneId;

  // Parse predicate operand.
  if (parser.parseLParen() || parser.parseRegionArgument(laneId) ||
      parser.parseRParen())
    return failure();

  if (parser.resolveOperand(laneId, builder.getIndexType(), result.operands))
    return failure();

  llvm::SMLoc inputsOperandsLoc;
  SmallVector<OpAsmParser::OperandType> inputsOperands;
  SmallVector<Type> inputTypes;
  if (succeeded(parser.parseOptionalKeyword("args"))) {
    if (parser.parseLParen())
      return failure();

    inputsOperandsLoc = parser.getCurrentLocation();
    if (parser.parseOperandList(inputsOperands) ||
        parser.parseColonTypeList(inputTypes) || parser.parseRParen())
      return failure();
  }
  if (parser.resolveOperands(inputsOperands, inputTypes, inputsOperandsLoc,
                             result.operands))
    return failure();

  // Parse optional results type list.
  if (parser.parseOptionalArrowTypeList(result.types))
    return failure();
  // Parse the region.
  if (parser.parseRegion(*warpRegion, /*arguments=*/{},
                         /*argTypes=*/{}))
    return failure();
  WarpSingleLaneOp::ensureTerminator(*warpRegion, builder, result.location);

  // Parse the optional attribute list.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void WarpSingleLaneOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (index.hasValue()) {
    regions.push_back(RegionSuccessor(getResults()));
    return;
  }

  // The warp region is always executed
  regions.push_back(RegionSuccessor(&warpRegion()));
}

void WarpSingleLaneOp::build(OpBuilder &builder, OperationState &result,
                             Value laneId) {
  build(builder, result, /*resultTypes=*/llvm::None, laneId);
}

void WarpSingleLaneOp::build(OpBuilder &builder, OperationState &result,
                             TypeRange resultTypes, Value laneId) {
  result.addOperands(laneId);
  result.addTypes(resultTypes);

  OpBuilder::InsertionGuard guard(builder);
  Region *warpRegion = result.addRegion();
  builder.createBlock(warpRegion);
}

#define GET_OP_CLASSES
#include "Dialects/VectorExt/VectorExtOps.cpp.inc"

using namespace mlir;
using namespace mlir::vector_ext;
