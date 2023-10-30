#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "pybind11/pybind11.h"

#include "dialect/dialect.h"
#include "mlir_lowering.h"
#include "transform_ops/dialect_extension.h"

PYBIND11_MODULE(_mlirDialectsJasc, m) {
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