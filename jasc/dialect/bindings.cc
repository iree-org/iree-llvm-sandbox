//===-- bindings.cc - Python bindings for Jasc dialect ----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "pybind11/pybind11.h"

#include "dialect/dialect.h"
#include "mlir_lowering.h"
#include "transform_ops/dialect_extension.h"

PYBIND11_MODULE(bindings, m) {
  m.def(
      "register_and_load_dialect",
      [](MlirContext py_context) {
        mlir::MLIRContext *context = unwrap(py_context);
        mlir::DialectRegistry registry;
        registry.insert<jasc::JascDialect>();
        jasc::registerTransformDialectExtension(registry);
        context->appendDialectRegistry(registry);
        context->loadDialect<jasc::JascDialect>();
      },
      pybind11::arg("context") = pybind11::none());

  m.def("register_lowering_passes",
        []() { jasc::registerMLIRLoweringPasses(); });
}
