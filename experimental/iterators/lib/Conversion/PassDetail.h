//===-- PassDetail.h - Iterators pass class details -------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef EXPERIMENTAL_ITERATORS_LIB_CONVERSION_PASSDETAIL_H
#define EXPERIMENTAL_ITERATORS_LIB_CONVERSION_PASSDETAIL_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_CLASSES
#include "iterators/Conversion/Passes.h.inc"

} // namespace mlir

#endif // EXPERIMENTAL_ITERATORS_LIB_CONVERSION_PASSDETAIL_H
