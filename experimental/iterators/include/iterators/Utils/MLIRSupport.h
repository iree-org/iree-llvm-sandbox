//===-- MLIRSupport.h - Utils for dealing with MLIR -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_UTILS_MLIR_SUPPORT_H
#define ITERATORS_UTILS_MLIR_SUPPORT_H

#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeRange.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"

namespace mlir {
class NamedAttribute;
class OpBuilder;
} // namespace mlir

namespace iterators {
namespace utils {

mlir::scf::WhileOp
createWhileOp(mlir::OpBuilder &builder, mlir::Location loc,
              mlir::TypeRange resultTypes, mlir::ValueRange operands,
              llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                      mlir::Block::BlockArgListType)>
                  beforeBuilder,
              llvm::function_ref<void(mlir::OpBuilder &, mlir::Location,
                                      mlir::Block::BlockArgListType)>
                  afterBuilder,
              llvm::ArrayRef<mlir::NamedAttribute> attributes = {});

} // namespace utils
} // namespace iterators

#endif // ITERATORS_UTILS_MLIR_SUPPORT_H
