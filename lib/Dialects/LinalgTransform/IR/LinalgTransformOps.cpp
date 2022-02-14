//===-- LinalgTransformOps.cpp - Linalg Transform dialect -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgTransform/LinalgTransformOps.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>

#include "Dialects/LinalgTransform/LinalgTransformOpsDialect.cpp.inc"

using namespace mlir;
using namespace mlir::linalg;

void transform::LinalgTransformDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialects/LinalgTransform/LinalgTransformOps.cpp.inc"
      >();
}

void transform::ScopeOp::getSuccessorRegions(
    Optional<unsigned> index, ArrayRef<Attribute> operands,
    SmallVectorImpl<RegionSuccessor> &regions) {
  if (index)
    regions.emplace_back(getResults());
  else
    regions.emplace_back(&body());
}

static LogicalResult verifySequenceOp(transform::SequenceOp op) {
  WalkResult result = op.walk([](Operation *child) {
    for (OpResult result : child->getResults()) {
      if (llvm::hasNItemsOrLess(result.getUses(), 1))
        continue;
      InFlightDiagnostic diag = child->emitError()
                                << "result #" << result.getResultNumber()
                                << " has more than one use";
      for (OpOperand &use : result.getUses()) {
        diag.attachNote(use.getOwner()->getLoc())
            << "used here as operand #" << use.getOperandNumber();
      }
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

LogicalResult transform::TileOp::verify() {
  if (!sizes().empty() && scalarize_dyn_dims()) {
    return emitOpError() << sizesAttrName() << " and "
                         << scalarize_dyn_dimsAttrName()
                         << " attributes are mutually exclusive";
  }

  ArrayAttr transposes = transpose_paddings();
  for (Attribute attr : transposes) {
    SmallVector<int64_t> transpose = extractFromI64ArrayAttr(attr);
    auto sequence = llvm::seq<int64_t>(0, transpose.size());
    if (!std::is_permutation(sequence.begin(), sequence.end(),
                             transpose.begin(), transpose.end())) {
      return emitOpError()
             << "expects transpose paddings to be a permutation, found "
             << attr;
    }
  }
  return success();
}

ParseResult transform::VectorizeOp::parse(OpAsmParser &parser,
                                          OperationState &result) {
  auto operationType = pdl::OperationType::get(parser.getContext());
  OpAsmParser::OperandType target;
  OptionalParseResult parseResult = parser.parseOptionalOperand(target);
  if (parseResult.hasValue()) {
    if (parseResult.getValue().failed() ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.resolveOperand(target, operationType, result.operands) ||
        parser.addTypeToList(operationType, result.types)) {
      return failure();
    }
  } else {
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return failure();
    }
  }
  return success();
}

void transform::VectorizeOp::print(OpAsmPrinter &printer) {
  if (target())
    printer << " " << target() << " ";
  printer.printOptionalAttrDict(getOperation()->getAttrs());
}

#define GET_OP_CLASSES
#include "Dialects/LinalgTransform/LinalgTransformOps.cpp.inc"
