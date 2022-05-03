//===-- VectorExtOps.h - Vector Extension dialect ops ------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialect/VectorExt/VectorExtOps.h"

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

ParseResult mlir::vector_ext::PredicateOp::parse(OpAsmParser &parser,
                                                 OperationState &result) {
  // Create the regions for 'truePredicate'.
  result.regions.reserve(1);
  Region *truePredicateRegion = result.addRegion();

  auto &builder = parser.getBuilder();

  // Parse all the operands.
  OpAsmParser::UnresolvedOperand predicateMask;
  OpAsmParser::UnresolvedOperand incomingMask;
  SmallVector<OpAsmParser::UnresolvedOperand> indices;
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

void mlir::vector_ext::PredicateOp::print(OpAsmPrinter &p) {
  bool printBlockTerminators = false;

  p << "(" << predicateMask() << ", [" << indices() << "], " << incomingMask()
    << ") : " << predicateMask().getType();
  if (!results().empty()) {
    p << " -> (" << getResultTypes() << ")";
    // Print yield explicitly if the op defines values.
    printBlockTerminators = true;
  }
  p << " ";
  p.printRegion(truePredicateRegion(),
                /*printEntryBlockArgs=*/true,
                /*printBlockTerminators=*/printBlockTerminators);

  p.printOptionalAttrDict(getOperation()->getAttrs());
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

#define GET_OP_CLASSES
#include "Dialect/VectorExt/VectorExtOps.cpp.inc"

using namespace mlir;
using namespace mlir::vector_ext;
