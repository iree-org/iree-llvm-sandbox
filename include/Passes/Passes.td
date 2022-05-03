//===-- Passes.td - Driver pass definition file -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef IREE_LLVM_SANDBOX_PASSES
#define IREE_LLVM_SANDBOX_PASSES

include "mlir/Pass/PassBase.td"

def LinalgFuse : Pass<"linalg-fuse", "FuncOp"> {
  let summary = "Pass to tile and fuse linalg operations.";
  let constructor = "mlir::createLinalgFusePass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,

    // Tiling options.
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">,
    ListOption<"tileInterchange", "tile-interchange", "int64_t",
                "Tile loop interchange",
                "llvm::cl::ZeroOrMore">,
    Option<"pad", "pad", "bool", /*default=*/"false",
      "Pad the anchor op operands.">,
    ListOption<"paddingValues", "padding-values", "std::string",
               "Operand padding values."
               "llvm::cl::ZeroOrMore">,
    ListOption<"paddingDimensions", "padding-dimensions", "int64_t",
               "Operation iterator dimensions to pad."
               "llvm::cl::ZeroOrMore">,
    ListOption<"packPaddings", "pack-paddings", "int64_t",
               "Operand packing flags.",
               "llvm::cl::ZeroOrMore">,
    ListOption<"hoistPaddings", "hoist-paddings", "int64_t",
               "Hoist padding depths.",
               "llvm::cl::ZeroOrMore">,
    ListOption<"transposePaddings", "transpose-paddings", "std::string",
               "Transpose paddings.",
               "llvm::cl::ZeroOrMore">,
    // Vectorization options.
    Option<"vectorize", "vectorize", "bool", /*default=*/"false",
      "Rewrite the linalg op as a vector operation.">,
    Option<"vectorizePadding", "vectorize-padding", "bool", /*default=*/"false",
      "Rewrite all tensor.pad ops in the function to vector form.">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithmeticDialect", "::mlir::AffineDialect",
    "::mlir::linalg::LinalgDialect", "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect", "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect"
  ];
}

def LinalgFuseOutputIntoReduction
    : Pass<"linalg-fuse-fill-into-reduction", "FuncOp"> {
  let summary = "Pass to fuse output into tiled reductions.";
  let constructor = "mlir::createLinalgFuseOutputIntoReductionPass()";
  let dependentDialects = [
    "::mlir::arith::ArithmeticDialect", "::mlir::AffineDialect",
    "::mlir::linalg::LinalgDialect", "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect", "::mlir::tensor::TensorDialect"
  ];
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which linalg op within the func is the anchor to latch on.">,

    // Tiling options.
    ListOption<"tileSizes", "tile-sizes", "int64_t", "Tile sizes",
               "llvm::cl::ZeroOrMore">
  ];
}

def UnrollOneVectorOp : Pass<"unroll-one-vector-op", "FuncOp"> {
  let summary = "Pass to unroll a vector op to a target size.";
  let constructor = "mlir::createUnrollOneVectorOpPass()";
  let options = [
    Option<"anchorFuncOpName", "anchor-func", "std::string", /*default=*/"",
      "Which func op is the anchor to latch on.">,
    Option<"anchorOpName", "anchor-op", "std::string", /*default=*/"",
      "Which unique op within the func is the anchor to latch on.">,

    // UnrollOneVectorOp options.
    ListOption<"sourceShape", "source-shape", "int64_t", "Source vector shape",
               "llvm::cl::ZeroOrMore">,
    ListOption<"targetShape", "target-shape", "int64_t", "Target vector shape",
               "llvm::cl::ZeroOrMore">,
  ];
  let dependentDialects = [
    "::mlir::arith::ArithmeticDialect", "::mlir::AffineDialect",
    "::mlir::linalg::LinalgDialect", "::mlir::scf::SCFDialect",
    "::mlir::func::FuncDialect", "::mlir::tensor::TensorDialect",
    "::mlir::vector::VectorDialect"
  ];
}

#endif // IREE_LLVM_SANDBOX_PASSES
