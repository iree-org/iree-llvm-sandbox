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

def LinalgSingleTilingExpert
    : Pass<"linalg-single-tiling-expert-driver", "FuncOp"> {
  let summary = "Pass to drive transformations on Linalg on tensors.";
  let constructor = "mlir::createLinalgSingleTilingExpertPass()";
  let options = [
    // Func / op targeting options.
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
    ListOption<"peeledLoops", "peeled-loops", "int64_t", "Peeled loops",
               "llvm::cl::ZeroOrMore">,
    Option<"pad", "pad", "bool", /*default=*/"false",
      "Pad the anchor op operands.">,
    ListOption<"paddingValues", "padding-values", "std::string",
               "Operand padding values."
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
    Option<"scalarizeDynamicDims", "scalarize-dynamic-dims", "bool",
      /*default=*/"false", "Tile dynamic dimensions by 1.">,

    // Generalization options.
    Option<"generalize", "generalize", "bool", /*default=*/"false",
      "Convert named operations to their generic form.">,
    ListOption<"iteratorInterchange", "iterator-interchange", "int64_t",
               "Interator interchange.",
               "llvm::cl::ZeroOrMore">,

    // Decomposition options.
    // TODO: atm this is applied to all supported ops. If/when we need finer
    // control this should be exposed with an opName + filter and a proper
    // pattern.
    Option<"decomposeToLowerDimOp", "decompose-to-lower-dim", "bool",
      /*default=*/"false",
      "Convert named operations to lower-D named operations.">,

    // Vectorization options.
    Option<"vectorize", "vectorize", "bool", /*default=*/"false",
      "Rewrite the linalg op as a vector operation.">,
    Option<"vectorizeOnlyTiled", "vectorize-only-tiled", "bool",
      /*default=*/"false",
      "Rewrite the linalg op as a vector operation only if it's tiled.">,
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

def LinalgBufferizationDriver : Pass<"linalg-bufferization-driver", "ModuleOp"> {
  let dependentDialects = [
    "::mlir::bufferization::BufferizationDialect"
  ];
  let summary = "Run module-level comprehensive inplace bufferization.";
  let constructor = "mlir::createLinalgBufferizationDriverPass()";
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
