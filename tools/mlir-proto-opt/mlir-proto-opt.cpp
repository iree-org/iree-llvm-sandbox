//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "CAPI.h"
#include "Dialects/LinalgExt/LinalgExtDialect.h"
#include "Dialects/LinalgExt/Passes.h"
#include "Dialects/VectorExt/VectorExtDialect.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

namespace mlir {
namespace test_ext {
void registerTestVectorMaskingUtils();
}
} // namespace mlir

using namespace llvm;
using namespace mlir;
using namespace mlir::linalg;
using namespace mlir::test_ext;

#ifdef SANDBOX_ENABLE_IREE_DIALECTS
#include "iree-dialects/Dialect/Input/InputDialect.h"
#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree-dialects/Dialect/LinalgExt/Transforms/Passes.h"

static void registerIreeDialects(DialectRegistry &registry) {
  registry.insert<mlir::iree_compiler::IREE::Input::IREEInputDialect>();
  registry.insert<mlir::iree_compiler::IREE::LinalgExt::IREELinalgExtDialect>();
  mlir::iree_compiler::IREE::LinalgExt::registerPasses();
}
#else
static void registerIreeDialects(DialectRegistry &registry) {}
#endif

void registerTestPasses() { registerTestVectorMaskingUtils(); }

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  registerAllPasses();
  ireeLlvmSandboxRegisterPasses();
  linalg_ext::registerLinalgExtPasses();
  registerTestPasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerIreeDialects(registry);
  registry.insert<linalg_ext::LinalgExtDialect, vector_ext::VectorExtDialect>();
  linalg_ext::registerTilingInterfaceExternalModels(registry);

  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
