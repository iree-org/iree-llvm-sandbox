//===-- RuntimeUtils.h - Utils for MLIR / C++ interop -----------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ITERATORS_UTILS_RUNTIMEUTILS_H
#define ITERATORS_UTILS_RUNTIMEUTILS_H

#ifdef _WIN32
#ifndef RUNTIME_UTILS_EXPORT
#ifdef runtime_utils_EXPORTS
// We are building this library
#define RUNTIME_UTILS_EXPORT __declspec(dllexport)
#else
// We are using this library
#define RUNTIME_UTILS_EXPORT __declspec(dllimport)
#endif // runtime_utils_EXPORTS
#endif // RUNTIME_UTILS_EXPORT
#else
// Non-windows: use visibility attributes.
#define RUNTIME_UTILS_EXPORT __attribute__((visibility("default")))
#endif // _WIN32

#include <cstdint>

//===---------------------------------------------------------------------===//
// Exposed C API.
//===---------------------------------------------------------------------===//

extern "C" RUNTIME_UTILS_EXPORT int8_t *iteratorsMakeSampleInputOperator();

extern "C" RUNTIME_UTILS_EXPORT void
iteratorsDestroySampleInputOperator(int8_t *op);

extern "C" RUNTIME_UTILS_EXPORT int8_t *
iteratorsMakeReduceOperator(int8_t *upstream);

extern "C" RUNTIME_UTILS_EXPORT void iteratorsDestroyReduceOperator(int8_t *op);

extern "C" RUNTIME_UTILS_EXPORT void iteratorsComsumeAndPrint(int8_t *upstream);

#endif // ITERATORS_UTILS_RUNTIMEUTILS_H
