//===-- Tabular.cpp - Tabular dialect ---------------------------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tabular/IR/Tabular.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::iterators;

//===----------------------------------------------------------------------===//
// Tabular dialect
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tabular/IR/TabularOpsDialect.cpp.inc"

void TabularDialect::initialize() {
#define GET_OP_LIST
  addOperations<
#include "iterators/Dialect/Tabular/IR/TabularOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "iterators/Dialect/Tabular/IR/TabularOpsTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Tabular interfaces
//===----------------------------------------------------------------------===//

#include "iterators/Dialect/Tabular/IR/TabularOpInterfaces.cpp.inc"
#include "iterators/Dialect/Tabular/IR/TabularTypeInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Tabular operations
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOps.cpp.inc"

LogicalResult ViewAsTabularOp::verify() {
  auto viewType = view().getType().cast<TabularViewType>();
  TypeRange columnTypes = viewType.getColumnTypes();

  // Verify matching column/element types.
  if (columnTypes.size() != memrefs().size()) {
    return emitOpError()
           << "type mismatch: should return a tabular view with the same "
           << "number of columns as the number of input memrefs (expected: "
           << memrefs().size() << ", found: " << columnTypes.size() << ").";
  }
  for (size_t i = 0; i < columnTypes.size(); i++) {
    Type memrefElementType =
        memrefs().getTypes()[i].cast<MemRefType>().getElementType();
    if (memrefElementType != columnTypes[i]) {
      return emitOpError()
             << "type mismatch: returned tabular view has column type "
             << columnTypes[i] << " at index " << i << " but should have type "
             << memrefElementType << ", the element type of the memref at the "
             << "same index.";
    }
  }

  // Verify all memrefs are of equal static length.
  if (!llvm::is_splat(llvm::map_range(memrefs().getTypes(), [](Type t) {
        return t.cast<MemRefType>().getDimSize(0);
      }))) {
    std::string lengths;
    {
      llvm::raw_string_ostream stream(lengths);
      llvm::interleaveComma(memrefs().getTypes(), stream, [&](Type type) {
        stream << type.cast<MemRefType>().getDimSize(0);
      });
    }
    return emitOpError()
           << "type mismatch: input memrefs cannot have different static "
           << "shapes (sizes found for dimension 0: " << lengths << ").";
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Tabular types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOpsTypes.cpp.inc"
