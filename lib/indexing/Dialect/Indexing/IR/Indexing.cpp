//===-- Indexing.cpp - Indexing dialect -----------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "indexing/Dialect/Indexing/IR/Indexing.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::indexing;

//===----------------------------------------------------------------------===//
// Indexing dialect
//===----------------------------------------------------------------------===//

#include "indexing/Dialect/Indexing/IR/IndexingOpsDialect.cpp.inc"

void IndexingDialect::initialize() {
#define GET_OP_LIST
    addOperations<

#include "indexing/Dialect/Indexing/IR/IndexingOps.cpp.inc"

    >();
    addTypes<
#define GET_TYPEDEF_LIST

#include "indexing/Dialect/Indexing/IR/IndexingOpsTypes.cpp.inc"

    >();
}

//===----------------------------------------------------------------------===//
// Indexing operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES

#include "indexing/Dialect/Indexing/IR/IndexingOps.cpp.inc"

LogicalResult mlir::indexing::GatherOp::inferReturnTypes(
        MLIRContext *context, std::optional<Location> location, ValueRange operands,
        DictionaryAttr attributes, RegionRange regions,
        SmallVectorImpl<Type> &inferredReturnTypes) {

    ArrayRef<int64_t> coordinates =
            attributes.get("coordinates").cast<mlir::DenseI64ArrayAttr>();
    RankedTensorType expectedResultType = mlir::tensor::GatherOp::inferResultType(
            // source
            operands[0].getType().cast<RankedTensorType>(),
            // indices
            operands[1].getType().cast<RankedTensorType>(), coordinates,
            /*rankReduced=*/true);
    inferredReturnTypes.assign({expectedResultType});
    return success();
}

//===----------------------------------------------------------------------===//
// Indexing types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES

#include "indexing/Dialect/Indexing/IR/IndexingOpsTypes.cpp.inc"
