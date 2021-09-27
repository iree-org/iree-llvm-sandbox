//===-- Passes.td - Runners pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef RUNNERS_PASSES
#define RUNNERS_PASSES

include "mlir/Pass/PassBase.td"

def LinalgTensorCodegenDriver
    : Pass<"linalg-tensor-codegen-driver", "ModuleOp"> {
  let summary = "Pass to drive transformations on Linalg on tensors.";
  let constructor = "mlir::createLinalgTensorCodegenDriverPass()";
  let options = [
    // Func / op targeting options.
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,

    // Tiling options.
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"tileInterchange", "tile-interchange", "int64_t",
                "Tile loop interchange",
                "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    ListOption<"peeledLoops", "peeled-loops", "int64_t", "Peeled loops",
               "llvm::cl::ZeroOrMore, llvm::cl::MiscFlags::CommaSeparated">,
    Option<"pad", "pad", "bool", /*default=*/"false",
      "Apply padding when tiling to ensure static data tile sizes.">,
    Option<"hoistPadding", "hoist-padding", "int", /*default=*/"0",
      "Hoist padding operations by the number of specified loops.">,
    Option<"scalarizeDynamicDims", "scalarize-dynamic-dims", "bool",
      /*default=*/"false", "Tile dynamic dimensions by 1.">,

    // Fusion options.
    Option<"fuse", "fuse", "bool", /*default=*/"false",
      "Rewrite the linalg op as a vector operation.">,

    // Vectorization options.
    Option<"vectorize", "vectorize", "bool", /*default=*/"false",
      "Rewrite the linalg op as a vector operation.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", /*default=*/"false",
      "Rewrite all linalg.pad_tensor ops in the function to vector form.">,

    // Bufferization options.
    Option<"bufferize", "bufferize", "bool", /*default=*/"false",
      "Run module-level comprehensive inplace bufferization.">,

    // Vector lowering options.
    Option<"vectorLowering", "lower-vector", "bool", /*default=*/"false",
      "Run transformations that lower high-level vectors.">,
    Option<"splitVectorTransfersTo", "split-transfers", "std::string",
      /*default=*/"",
      [{Split vector transfers between slow (masked) and fast "
        "(unmasked) variants. Possible options are:\n"
          "\tnone [default]: keep unsplit vector.transfer and pay the price\n"
          "\tlinalg-copy: use linalg.fill + linalg.copy for the slow path\n"
          "\tvector-transfers: use extra small unmasked vector.transfers for"
          " the slow path\n}]>,
    Option<"vectorizeContractionTo", "vectorize-contraction-to", "std::string",
      /*default=*/[{"outerproduct"}],
      [{Lower vector.contract to finer-grained vector ops, options are:\n"
          "\touterproduct [default]\n"
          "\tdot\n"
          "\tmatrixintrinsics\n}]>,
    Option<"unrollVectorTransfers", "unroll-vector-transfers", "bool",
      /*default=*/"true",
      "Run transformations that lower high-level vectors.">,

    // LLVM lowering options.
    Option<"llvmLowering", "lower-to-llvm", "bool", /*default=*/"false",
      "Run transformations and passes to produce LLVM IR.">,
  ];
  let dependentDialects = [
    "AffineDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "StandardOpsDialect",
    "tensor::TensorDialect",
    "vector::VectorDialect"
  ];
}

#endif // RUNNERS_PASSES
