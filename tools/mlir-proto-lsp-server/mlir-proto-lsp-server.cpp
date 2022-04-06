//===-- mlir-proto-lsp-server.cpp - LSP server for sandbox ------*- C++ -*-===//
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

#include "Registration.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

using namespace mlir;

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registerAllPasses();
  registerIntoDialectRegistry(registry);
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
