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

LogicalResult CreateTabularViewOp::verify() {
  auto viewType = view().getType().cast<TabularViewType>();
  TypeRange columnTypes = viewType.getColumnTypes();

  // Verify matching column/element types.
  if (!llvm::all_of_zip(memrefs().getTypes(), columnTypes,
                        [](Type t1, Type t2) {
                          return t1.cast<MemRefType>().getElementType() == t2;
                        })) {

    std::string viewColumnTypes;
    {
      llvm::raw_string_ostream stream(viewColumnTypes);
      llvm::interleaveComma(columnTypes, stream,
                            [&](Type type) { type.print(stream); });
    }
    std::string memrefColumnTypes;
    {
      llvm::raw_string_ostream stream(memrefColumnTypes);
      llvm::interleaveComma(memrefs().getTypes(), stream, [&](Type type) {
        type.cast<MemRefType>().getElementType().print(stream);
      });
    }
    return emitOpError()
           << "type mismatch: should return a tabular view with the column "
           << "types of the elements of the input memrefs (namely: "
           << "tabular_view<" << memrefColumnTypes << ">) "
           << "but returns a tabular_view<" << viewColumnTypes << ">.";
  }

  // Verify memrefs have rank 1 and contiguous memory layout.
  {
    for (const auto &indexedMemref : llvm::enumerate(memrefs().getTypes())) {
      // Must have static rank = 1.
      auto memrefType = indexedMemref.value().cast<MemRefType>();
      if (memrefType.getRank() != 1)
        return emitOpError()
               << "unsupported type: input memref #" << indexedMemref.index()
               << " has rank != 1, which is not supported.";

      // Only support contiguous memory layout.
      if (!isStaticShapeAndContiguousRowMajor(memrefType))
        return emitOpError()
               << "unsupported type: input memref #" << indexedMemref.index()
               << " does not have a static shape and a contiguous memory "
                  "layout, both of which are required.";
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
                "shapes (sizes found for dimension 0: "
             << lengths << ").";
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Tabular types
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "iterators/Dialect/Tabular/IR/TabularOpsTypes.cpp.inc"
