//===-- iterators-lsp-server.cpp - LSP server for Iterators -----*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a sandbox-specific MLIR LSP Language server. This
/// extends the as-you-type diagnostics in VS Code to dialects defined in the
/// sandbox. Implementation essentially as explained here:
/// https://mlir.llvm.org/docs/Tools/MLIRLSP/.
///
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "structured/Conversion/Passes.h"
#include "structured/Dialect/Iterators/IR/Iterators.h"
#include "structured/Dialect/Iterators/Transforms/Passes.h"
#include "structured/Dialect/Tabular/IR/Tabular.h"
#include "structured/Dialect/Tuple/IR/Tuple.h"
#include "structured/Dialect/Tuple/Transforms/Passes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Signals.h"

using namespace mlir;

static void registerIteratorDialects(DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      mlir::iterators::IteratorsDialect,
      mlir::tabular::TabularDialect,
      mlir::tuple::TupleDialect
      // clang-format on
      >();
}

int main(int argc, char **argv) {
#ifndef NDEBUG
  static std::string executable =
      llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  llvm::sys::PrintStackTraceOnErrorSignal(executable);
#endif

  registerAllPasses();
  registerStructuredConversionPasses();
  registerIteratorsPasses();
  registerTuplePasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);
  registerIteratorDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
