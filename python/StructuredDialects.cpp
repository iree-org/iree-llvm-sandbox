//===- IteratorsExtension.cpp - Extension module --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <mlir-c/BuiltinAttributes.h>
#include <pybind11/pybind11.h>

#include <string>

#include "mlir-c/IR.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "structured-c/Dialects.h"
#include <mlir-c/BuiltinTypes.h>
#include <vector>

namespace py = pybind11;
using namespace mlir::python::adaptors;

static MlirStringRef toMlirStringRef(const std::string &s) {
  return mlirStringRefCreate(s.data(), s.size());
}

namespace {
// TODO(max): this is plucked from PybindUtils.cpp upstream - refactor upstream
pybind11::error_already_set SetPyError(PyObject *excClass,
                                       const llvm::Twine &message) {
  auto messageStr = message.str();
  PyErr_SetString(excClass, messageStr.c_str());
  return pybind11::error_already_set();
}

} // namespace

PYBIND11_MODULE(_structuredDialects, mainModule) {
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

  mlir_type_subclass(iteratorsModule, "StreamType",
                     mlirTypeIsAIteratorsStreamType)
      .def_classmethod(
          "get",
          [](const py::object &cls, MlirType elementType, MlirContext context) {
            return cls(mlirIteratorsStreamTypeGet(context, elementType));
          },
          py::arg("cls"), py::arg("element_type"),
          py::arg("context") = py::none());

  //===--------------------------------------------------------------------===//
  // Tabular dialect.
  //===--------------------------------------------------------------------===//
  auto tabularModule = mainModule.def_submodule("tabular");

  //
  // Dialect
  //

  tabularModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__tabular__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //
  // Types
  //

  mlir_type_subclass(tabularModule, "TabularViewType", mlirTypeIsATabularView)
      .def_classmethod(
          "get",
          [](const py::object &cls, const py::list &columnTypeList,
             MlirContext context) {
            intptr_t num = py::len(columnTypeList);
            // Mapping py::list to SmallVector.
            llvm::SmallVector<MlirType, 4> columnTypes;
            for (auto columnType : columnTypeList) {
              columnTypes.push_back(columnType.cast<MlirType>());
            }
            return cls(
                mlirTabularViewTypeGet(context, num, columnTypes.data()));
          },
          py::arg("cls"), py::arg("column_types"),
          py::arg("context") = py::none())
      .def("get_column_type", mlirTabularViewTypeGetColumnType, py::arg("pos"))
      .def("get_num_column_types", mlirTabularViewTypeGetNumColumnTypes)
      .def("get_row_type", mlirTabularViewTypeGetRowType);

  //===--------------------------------------------------------------------===//
  // Tuple dialect.
  //===--------------------------------------------------------------------===//
  auto tupleModule = mainModule.def_submodule("tuple");

  //
  // Dialect
  //

  tupleModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__tuple__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //===--------------------------------------------------------------------===//
  // Indexing dialect.
  //===--------------------------------------------------------------------===//
  auto indexingModule = mainModule.def_submodule("indexing");

  //
  // Dialect
  //

  indexingModule.def(
      "register_dialect",
      [](MlirContext context, bool doLoad) {
        MlirDialectHandle handle = mlirGetDialectHandle__indexing__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (doLoad) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);

  //
  // Types
  //

  (void)mlir_value_subclass(indexingModule, "TensorValue", [](MlirValue value) {
    return mlirIsATensorValue(value);
  });

  mlir_type_subclass(indexingModule, "IndexTensorType",
                     [](MlirType type) {
                       return mlirTypeIsATensor(type) &&
                              mlirTypeIsAIndex(
                                  mlirShapedTypeGetElementType(type));
                     })
      .def_classmethod(
          "get",
          [](const py::object &cls, const std::vector<int64_t> &shape,
             MlirContext context) {
            return cls(mlirRankedTensorTypeGet(shape.size(), shape.data(),
                                               mlirIndexTypeGet(context),
                                               mlirAttributeGetNull()));
          },
          py::arg("cls"), py::arg("value"), py::arg("context") = py::none());

  (void)mlir_value_subclass(indexingModule, "ArithValue", [](MlirValue value) {
    return mlirIsAnArithValue(value);
  });
}