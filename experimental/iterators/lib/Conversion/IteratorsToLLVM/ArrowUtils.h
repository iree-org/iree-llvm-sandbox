//===-- ArrowUtils.h - Utils for converting Arrow to LLVM -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CONVERSION_ITERATORSTOLLVM_ARROWUTILS_H
#define LIB_CONVERSION_ITERATORSTOLLVM_ARROWUTILS_H

namespace mlir {
class ModuleOp;
class Type;
namespace LLVM {
class LLVMFuncOp;
} // namespace LLVM
} // namespace mlir

namespace mlir {
namespace iterators {

/// Ensures that the runtime function `mlirIteratorsArrowArrayGetSize` is
/// present in the current module and returns the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp lookupOrInsertArrowArrayGetSize(mlir::ModuleOp module);

/// Ensures that the runtime function `mlirIteratorsArrowArrayGet*Column`
/// corresponding to the given type is present in the current module and returns
/// the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp
lookupOrInsertArrowArrayGetColumn(mlir::ModuleOp module,
                                  mlir::Type elementType);

/// Ensures that the runtime function `mlirIteratorsArrowArrayRelease` is
/// present in the current module and returns the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp lookupOrInsertArrowArrayRelease(mlir::ModuleOp module);

/// Ensures that the runtime function `mlirIteratorsArrowSchemaRelease` is
/// present in the current module and returns the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp lookupOrInsertArrowSchemaRelease(mlir::ModuleOp module);

/// Ensures that the runtime function `mlirIteratorsArrowArrayStreamGetSchema`
/// is present in the current module and returns the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp
lookupOrInsertArrowArrayStreamGetSchema(mlir::ModuleOp module);

/// Ensures that the runtime function `mlirIteratorsArrowArrayStreamGetNext` is
/// present in the current module and returns the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp
lookupOrInsertArrowArrayStreamGetNext(mlir::ModuleOp module);

/// Ensures that the runtime function `mlirIteratorsArrowArrayStreamRelease` is
/// present in the current module and returns the corresponding LLVM func op.
mlir::LLVM::LLVMFuncOp
lookupOrInsertArrowArrayStreamRelease(mlir::ModuleOp module);

} // namespace iterators
} // namespace mlir

#endif // LIB_CONVERSION_ITERATORSTOLLVM_ARROWUTILS_H
