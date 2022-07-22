//===- IteratorsExtension.cpp - Extension module --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <pybind11/pybind11.h>

#include <array>
#include <initializer_list>

#include "iterators-c/Dialects.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

namespace py = pybind11;
using namespace mlir::python::adaptors;

/// Sets a python error, ready to be thrown to return control back to the
/// python runtime.
/// Correct usage:
///   throw SetPyError(PyExc_ValueError, "Foobar'd");
// Copied from llvm-project/mlir/lib/Bindings/Python/PybindUtils.cpp.
py::error_already_set setPyError(PyObject *excClass,
                                 const llvm::Twine &message) {
  auto messageStr = message.str();
  PyErr_SetString(excClass, messageStr.c_str());
  return py::error_already_set();
}

PYBIND11_MODULE(_iteratorsDialects, mainModule) {
  //===--------------------------------------------------------------------===//
  // Iterators dialect
  //===--------------------------------------------------------------------===//
  auto iteratorsModule = mainModule.def_submodule("iterators");

  //
  // Dialect
  //

  iteratorsModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__iterators__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //
  // Types
  //

  mlir_type_subclass(iteratorsModule, "ColumnarBatchType",
                     mlirTypeIsAIteratorsColumnarBatchType)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirType elementType, MlirContext context) {
            if (!mlirTypeIsATuple(elementType)) {
              throw setPyError(
                  PyExc_ValueError,
                  llvm::Twine(
                      "invalid element_type: must be TupleType, found '") +
                      py::repr(py::cast(elementType)).cast<std::string>() +
                      "'.");
            }
            return cls(mlirIteratorsColumnarBatchTypeGet(context, elementType));
          },
          py::arg("cls"), py::arg("element_type"),
          py::arg("context") = py::none());

  mlir_type_subclass(iteratorsModule, "StreamType",
                     mlirTypeIsAIteratorsStreamType)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirType elementType, MlirContext context) {
            return cls(mlirIteratorsStreamTypeGet(context, elementType));
          },
          py::arg("cls"), py::arg("element_type"),
          py::arg("context") = py::none());
}
