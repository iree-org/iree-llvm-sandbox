//===- IRTypes.cpp - Indexing ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "IRTypes.h"
#include "indexing/Dialect/Indexing/IR/Indexing.h"

#include "IRModule.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::indexing;

bool mlirTypeIsACustom(MlirType type) { return unwrap(type).isa<CustomType>(); }

MlirType mlirCustomTypeGet(MlirContext ctx, MlirStringRef value) {
    return wrap(CustomType::get(unwrap(ctx), unwrap(value)));
}

static MlirStringRef toMlirStringRef(const std::string &s) {
    return mlirStringRefCreate(s.data(), s.size());
}

MlirStringRef mlirCustomTypeGetValue(MlirType type) {
    return wrap(unwrap(type).cast<CustomType>().getValue());
}

/// Custom Type subclass - CustomType.
class PyCustomType : public PyConcreteType<PyCustomType> {
public:
    static constexpr IsAFunctionTy isaFunction = mlirTypeIsACustom;
    static constexpr const char *pyClassName = "CustomType";
    using PyConcreteType::PyConcreteType;

    static void bindDerived(ClassTy &c) {
        c.def_static(
                "get",
                [](const std::string &value, DefaultingPyMlirContext context) {
                    MlirType type =
                            mlirCustomTypeGet(context->get(), toMlirStringRef(value));
                    return PyCustomType(context->getRef(), type);
                },
                py::arg("value"), py::arg("context") = py::none(),
                "Create an indexing dialect Custom type.");
        c.def_property_readonly(
                "value",
                [](PyCustomType &self) {
                    MlirStringRef stringRef = mlirCustomTypeGetValue(self);
                    return py::str(stringRef.data, stringRef.length);
                },
                "Returns the value for the Custom type as a string.");
    }
};

void mlir::indexing::populateIRTypes(py::module &m) { PyCustomType::bind(m); }