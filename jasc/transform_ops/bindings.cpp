#include "mlir/CAPI/IR.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/lib/Bindings/Python/IRModule.h"
#include "pybind11/attr.h"
#include "pybind11/pybind11.h"

#include "dialect_extension.h"

PYBIND11_MODULE(_mlirTransformOpsJasc, m) {
  m.def(
      "register_transform_dialect_extension",
      [](mlir::python::DefaultingPyMlirContext py_context) {
        mlir::MLIRContext *context = unwrap(py_context->get());
        mlir::DialectRegistry registry;
        jasc::registerTransformDialectExtension(registry);
        context->appendDialectRegistry(registry);
      },
      "context"_a = pybind11::none());
}
