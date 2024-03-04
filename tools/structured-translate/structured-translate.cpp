//===-- structured-translate.cpp - "structured" mlir-translate --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// `mlir-stranslate` with translations to and from "structured" dialects, i.e.,
// dialects from this repository.
//===----------------------------------------------------------------------===//

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "structured/Dialect/Substrait/IR/Substrait.h"
#include "structured/Target/SubstraitPB/Export.h"
#include "structured/Target/SubstraitPB/Import.h"
#include "llvm/Support/GraphWriter.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace substrait {

static void registerSubstraitDialects(DialectRegistry &registry) {
  registry.insert<mlir::substrait::SubstraitDialect>();
}

void registerSubstraitToProtobufTranslation() {
  TranslateFromMLIRRegistration registration(
      "substrait-to-protobuf", "translate from Substrait MLIR to protobuf",
      translateSubstraitToProtobuf, registerSubstraitDialects);
}

void registerProtobufToSubstraitTranslation() {
  TranslateToMLIRRegistration registration(
      "protobuf-to-substrait", "translate from protobuf to Substrait MLIR",
      translateProtobufToSubstrait, registerSubstraitDialects);
}

} // namespace substrait
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllTranslations();
  mlir::substrait::registerSubstraitToProtobufTranslation();
  mlir::substrait::registerProtobufToSubstraitTranslation();

  return failed(
      mlir::mlirTranslateMain(argc, argv, "MLIR Translation Testing Tool"));
}
