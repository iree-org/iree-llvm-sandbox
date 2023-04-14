#include "IRTypes.h"

#include "indexing/Dialect/Indexing/IR/Indexing.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Wrap.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/FileSystem.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::indexing;

void registerIndexing(MLIRContext &context) {
    DialectRegistry registry;
    registry.insert<mlir::indexing::IndexingDialect>();
    context.appendDialectRegistry(registry);
}

PYBIND11_MODULE(_indexingDialects, m) {
    m.def(
            "register_dialect",
            [](const py::handle mlirContext) {
                auto *context = unwrap(mlirPythonCapsuleToContext(
                        py::detail::mlirApiObjectToCapsule(mlirContext).ptr()));
                registerIndexing(*context);
                context->getOrLoadDialect<mlir::indexing::IndexingDialect>();
            },
            py::arg("context") = py::none());

    indexing::populateIRTypes(m);
}
