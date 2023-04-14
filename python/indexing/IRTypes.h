//===- IRTypes.h - Indexing ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INDEXING_IRTYPES_H
#define INDEXING_IRTYPES_H

#include <pybind11/pybind11.h>

#include "indexing/Dialect/Indexing/IR/Indexing.h"

namespace py = pybind11;

namespace mlir::indexing {
    void populateIRTypes(py::module &m);
}

#endif // INDEXING_IRTYPES_H
