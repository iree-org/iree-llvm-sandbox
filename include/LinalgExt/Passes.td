//===-- Passes.td - Linalg Ext transformation passes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef RUNNERS_LINALGEXT_PASSES
#define  RUNNERS_LINALGEXT_PASSES

include "mlir/Pass/PassBase.td"

def LinalgExtTiling : Pass<"linalg-ext-tiling", "FuncOp"> {
  let summary = "Pass to tiling linalg_ext operations.";
  let constructor = "mlir::linalg_ext::createLinalgExtTilingPass()";
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "tensor::TensorDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile Sizes",
    "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">
  ];
}

def LinalgExtToSCF : Pass<"lower-linalg-ext-to-scf", "FuncOp"> {
  let summary = "Pass to lower linalg_ext.tile to scf.for.";
  let constructor = "mlir::linalg_ext::createLinalgExtToSCFPass()";
  let dependentDialects = [
    "AffineDialect",
    "tensor::TensorDialect",
    "scf::SCFDialect"
  ];
}

#endif // RUNNERS_LINALGEXT_PASSES
