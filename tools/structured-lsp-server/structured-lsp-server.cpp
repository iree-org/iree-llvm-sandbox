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
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "structured/Conversion/Passes.h"
#include "structured/Dialect/Iterators/IR/Iterators.h"
#include "structured/Dialect/Iterators/Transforms/Passes.h"
#include "structured/Dialect/Tabular/IR/Tabular.h"
#include "structured/Dialect/Tuple/IR/Tuple.h"
#include "structured/Dialect/Tuple/Transforms/Passes.h"

using namespace mlir;

static void registerIteratorDialects(DialectRegistry &registry) {
  registry.insert<
      // clang-format off
      mlir::structured::IteratorsDialect,
      mlir::structured::TabularDialect,
      mlir::structured::TupleDialect
      // clang-format on
      >();
}

int main(int argc, char **argv) {
  registerAllPasses();
  registerStructuredConversionPasses();
  registerIteratorsPasses();
  registerTuplePasses();

  DialectRegistry registry;
  registerAllDialects(registry);
  registerIteratorDialects(registry);

  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
