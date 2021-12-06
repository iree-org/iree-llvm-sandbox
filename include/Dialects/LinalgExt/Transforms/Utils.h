//===- Utils.h - Utils for simplifying writing transformations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_DIALECTS_LINALGEXT_TRANSFORMS_UTILS_H_
#define IREE_LLVM_SANDBOX_DIALECTS_LINALGEXT_TRANSFORMS_UTILS_H_

namespace mlir {
class Location;
class OpBuilder;
class Value;

namespace tensor {
class ExtractSliceOp;
}

namespace linalg_ext {

/// Insert the `source` tensor into the `dest` tensor by creating the relevant
/// `subset_insert` op. The details of the `subset_insert` op are retrieved
/// from the `subset_extract` op so that they form a matching extract/insert
/// pair.
Value createMatchingSubsetInsertOp(OpBuilder &b, Location loc,
                                   tensor::ExtractSliceOp subsetExtractOp,
                                   Value source, Value dest);

} // namespace linalg_ext
} // namespace mlir

#endif // IREE_LLVM_SANDBOX_DIALECTS_LINALGEXT_TRANSFORMS_UTILS_H_
