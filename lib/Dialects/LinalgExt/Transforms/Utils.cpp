//===- Utils.cpp - LinalgExt transform utils ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Dialects/LinalgExt/Transforms/Utils.h"
#include "Dialects/LinalgExt/LinalgExtOps.h"
#include "Dialects/LinalgExt/PassDetail.h"
#include "Dialects/LinalgExt/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Identifier.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::linalg_ext;

/// Insert the `source` tensor into the `dest` tensor by creating the relevant
/// `subset_insert` op. The details of the `subset_insert` op are retrieved
/// from the `subset_extract` op so that they form a matching extract/insert
/// pair.
Value mlir::linalg_ext::createMatchingSubsetInsertOp(
    OpBuilder &b, Location loc, tensor::ExtractSliceOp subsetExtractOp,
    Value source, Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, subsetExtractOp.source().getType(), source, dest,
      subsetExtractOp.offsets(), subsetExtractOp.sizes(),
      subsetExtractOp.strides(), subsetExtractOp.static_offsets(),
      subsetExtractOp.static_sizes(), subsetExtractOp.static_strides());
}
