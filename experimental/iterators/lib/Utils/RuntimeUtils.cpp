//===- RuntimeUtils.cpp - Utils for MLIR / C++ interop -------------------===//

#include "iterators/Utils/RuntimeUtils.h"

#include <cstdio>

//===---------------------------------------------------------------------===//
// Exposed C API.
//===---------------------------------------------------------------------===//
extern "C" int64_t oneWay() {
  void *p = reinterpret_cast<void *>(0xDEADBEEF);
  fprintf(stdout, "Create a dummy pointer %p\n", p);
  return reinterpret_cast<int64_t>(p);
}

extern "C" void otherWay(int64_t p) {
  fprintf(stdout, "Use a dummy pointer %p\n", p);
}