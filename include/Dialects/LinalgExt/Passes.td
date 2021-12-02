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

def InParallelToSequentialFor : Pass<"linalg-in-parallel-to-sequential-for", "FuncOp"> {
  let summary = "Pass to lower linalg_ext.in_parallel to scf.for.";
  let description = [{
    Rewrite linalg_ext.in_parallel to a sequential scf.for.
    
    Warning: when the linalg_ext.in_parallel terminator operates on tensors, 
    this is a potentially dangerous transformation under the current semantics.
    In order for this transformation to be semantics-preserving, 2 conditions need
    to come together that are currently not checked and the subject of future analyses:
      1. The terminator of linalg_ext.in_parallel may compose the output tensor in 
      potentially undefined ways: if the linalg_ext.parallel_insert_slice regions overlap, they 
      may occur in any order and the result is unspecified. A future overlap/intersection
      analysis will be needed to guard against this case.
      2. Even when condition 1. has well-defined behavior, semantics altering behavior may 
      still be introduced to simplify inplace bufferization. In the current implementation, 
      all linalg_ext.parallel_insert_slice dest operands are optimistically turned into scf.for
      iter_args. This is optimistic because any use of a tensor inside linalg_ext.in_parallel
      is guaranteed to see the value before the start of the op; whereas the same use may 
      see a partially updated sequential result in the scf.for op.
      An extra analysis is needed to ensure that a particular read of a result tensor would 
      see the initial value and not a partial update. This is guaranteed by construction when 
      the linalg_ext.in_parallel op is produced by lowering a linalg_ext.tile operation but
      is not something that is generally enforced in the IR.
      For the moment we perform the replacement of the use with the scf.for iter_arg to be
      able to connect pieces inplace with bufferization. In the future an analysis will be 
      needed to ensure correctness of this lowering to sequential scf.for + iter_args.
  }];
  let constructor = "mlir::linalg_ext::createInParallelToSequentialForPass()";
  let dependentDialects = [
    "AffineDialect",
    "tensor::TensorDialect",
    "scf::SCFDialect"
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
