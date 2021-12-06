//===- Utils.h - Utils for simplifying writing transformations -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_TRANSFORMS_UTILS_H_
#define IREE_LLVM_SANDBOX_TRANSFORMS_UTILS_H_

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/AffineExpr.h"

namespace mlir {
namespace linalg_ext {

struct AffineValueExpr {
  explicit AffineValueExpr(AffineExpr e) : e(e) {}
  AffineValueExpr bind(Value v) {
    this->v = v;
    return *this;
  }
  operator AffineExpr() const { return e; }
  operator Value() const { return v; }
  AffineExpr e;
  Value v;
};

/// Helper struct to build simple arithmetic quantiAffineValueExprs with minimal
/// type inference support.
// TODO: move into ArithBuilder once ops have been moved into arith.
struct AffineBuilder {
  AffineBuilder(OpBuilder &b, Location loc) : b(b), loc(loc) {}

  Value add(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e + rhs.e}, ValueRange{lhs, rhs});
  }
  Value sub(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e - rhs.e}, ValueRange{lhs, rhs});
  }
  Value mul(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e * rhs.e}, ValueRange{lhs, rhs});
  }
  Value ceil(AffineValueExpr lhs, AffineValueExpr rhs) {
    return b.createOrFold<AffineApplyOp>(
        loc, ArrayRef<AffineExpr>{lhs.e.ceilDiv(rhs.e)}, ValueRange{lhs, rhs});
  }
  Value min(ValueRange vals) {
    return b.createOrFold<AffineMinOp>(
        loc, AffineMap::getMultiDimIdentityMap(vals.size(), b.getContext()),
        vals);
  }
  Value max(ValueRange vals) {
    return b.createOrFold<AffineMinOp>(
        loc, AffineMap::getMultiDimIdentityMap(vals.size(), b.getContext()),
        vals);
  }

private:
  OpBuilder &b;
  Location loc;
};

} // namespace linalg_ext
} // namespace mlir

#endif // IREE_LLVM_SANDBOX_TRANSFORMS_UTILS_H_