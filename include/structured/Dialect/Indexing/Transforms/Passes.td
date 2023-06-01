//===-- Passes.td - Transform pass definition file ---------*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_TRANSFORMS_PASSES
#define INDEXING_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// GatherToExtractSlice
//===----------------------------------------------------------------------===//

def GatherToExtractSlice : Pass<"gather-to-extract-slice", "func::FuncOp"> {
  let summary = "Transform indexing::GatherOp to tensor::ExtractSliceOp";
  let constructor = "mlir::createGatherToExtractSlicePass()";
  let dependentDialects = [
    "tensor::TensorDialect",
    "func::FuncDialect",
  ];
}

//===----------------------------------------------------------------------===//
// ScatterToInsertSlice
//===----------------------------------------------------------------------===//

def ScatterToInsertSlice : Pass<"scatter-to-insert-slice", "func::FuncOp"> {
  let summary = "Transform indexing::ScatterOp to tensor::InsertSliceOp";
  let constructor = "mlir::createScatterToInsertSlicePass()";
  let dependentDialects = [
    "tensor::TensorDialect",
    "func::FuncDialect",
  ];
}

def MungeCallingConventions : Pass<"refbackend-munge-calling-conventions", "ModuleOp"> {
  let summary = "Munge calling conventions for calling via ExecutionEngine";
  let constructor = "mlir::createMungeCallingConventionsPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

#endif // INDEXING_TRANSFORMS_PASSES
