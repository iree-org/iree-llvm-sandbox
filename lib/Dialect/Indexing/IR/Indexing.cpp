//===-- Indexing.cpp - Indexing dialect -------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Indexing/IR/Indexing.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::indexing;

//===----------------------------------------------------------------------===//
// Indexing dialect
//===----------------------------------------------------------------------===//

#include "structured/Dialect/Indexing/IR/IndexingOpsDialect.cpp.inc"

void IndexingDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "structured/Dialect/Indexing/IR/IndexingOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "structured/Dialect/Indexing/IR/IndexingOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Indexing operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES

#include "structured/Dialect/Indexing/IR/IndexingOps.cpp.inc"

LogicalResult mlir::indexing::GatherOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  ArrayRef<int64_t> gather_dims =
      attributes.get("gather_dims").cast<mlir::DenseI64ArrayAttr>();
  RankedTensorType expectedResultType = mlir::tensor::GatherOp::inferResultType(
      // source
      operands[0].getType().cast<RankedTensorType>(),
      // indices
      operands[1].getType().cast<RankedTensorType>(), gather_dims,
      /*rankReduced=*/true);
  inferredReturnTypes.assign({expectedResultType});
  return success();
}

LogicalResult mlir::indexing::ConcatenateOp::inferReturnTypes(
    MLIRContext *context, std::optional<Location> location, ValueRange operands,
    DictionaryAttr attributes, OpaqueProperties properties, RegionRange regions,
    SmallVectorImpl<Type> &inferredReturnTypes) {

  auto dimension = attributes.get("dimension").cast<IntegerAttr>().getInt();
  auto sourceType = operands[0].getType().cast<RankedTensorType>();
  SmallVector<int64_t> resultShape(sourceType.getShape());
  std::for_each(
      operands.begin() + 1, operands.end(),
      [&resultShape, dimension](const Value &v) {
        resultShape[dimension] +=
            v.getType().cast<RankedTensorType>().getShape()[dimension];
      });
  inferredReturnTypes.assign(
      {RankedTensorType::Builder(sourceType).setShape(resultShape)});
  return success();
}

//===----------------------------------------------------------------------===//
// Indexing types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES

#include "structured/Dialect/Indexing/IR/IndexingOpsTypes.cpp.inc"
