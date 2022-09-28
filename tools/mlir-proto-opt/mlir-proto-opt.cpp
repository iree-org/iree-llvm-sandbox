//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
// Reuse Registration.cpp as much as possible.
//
//===----------------------------------------------------------------------===//

#include "Dialect/VectorExt/VectorExtDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;

#ifdef SANDBOX_ENABLE_ITERATORS
#include "iterators/Conversion/Passes.h"
#include "iterators/Dialect/Iterators/IR/Iterators.h"
#include "iterators/Dialect/Tabular/IR/Tabular.h"

static void registerIteratorDialects(DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      mlir::iterators::IteratorsDialect,
      mlir::iterators::TabularDialect
      // clang-format on
      >();
  registerIteratorsConversionPasses();
}
#else
static void registerIteratorDialects(DialectRegistry &registry) {}
#endif

#ifdef SANDBOX_ENABLE_ALP
#include "alp/Transforms/Passes.h"
static void registerALPPasses() { registerALPPasses(); }
#else
static void registerALPPasses() {}
#endif

namespace mlir {
namespace test_ext {
void registerTestVectorMaskingUtils();
} // namespace test_ext
} // namespace mlir

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  registerALPPasses();

  mlir::test_ext::registerTestVectorMaskingUtils();

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<vector_ext::VectorExtDialect>();
  registerIteratorDialects(registry);

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/true));
}
