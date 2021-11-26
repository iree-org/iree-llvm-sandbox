//===-- Passes.td - Linalg Ext transformation passes -------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef DIALECTS_LINALGEXT_PASSES
#define DIALECTS_LINALGEXT_PASSES

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

def TileToSequentialFor : Pass<"linalg-tile-to-sequential-for", "FuncOp"> {
  let summary = "Pass to lower linalg_ext.tile to scf.for.";
  let constructor = "mlir::linalg_ext::createTileToSequentialForPass()";
  let dependentDialects = [
    "AffineDialect",
    "tensor::TensorDialect",
    "scf::SCFDialect"
  ];
}

def TileToInParallel : Pass<"linalg-tile-to-in-parallel", "FuncOp"> {
  let summary = "Pass to lower linalg_ext.tile to linalg_ext.in_parallel.";
  let constructor = "mlir::linalg_ext::createTileToInParallelPass()";
  let dependentDialects = [
    "AffineDialect",
    "tensor::TensorDialect",
    "scf::SCFDialect"
  ];
}

#endif // DIALECTS_LINALGEXT_PASSES
