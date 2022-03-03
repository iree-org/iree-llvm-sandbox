//===- RuntimeUtils.h - Utils for MLIR / C++ interop ---------------------===//

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
extern "C" RUNTIME_UTILS_EXPORT int64_t oneWay();
extern "C" RUNTIME_UTILS_EXPORT void otherWay(int64_t);

#endif // ITERATORS_UTILS_RUNTIMEUTILS_H