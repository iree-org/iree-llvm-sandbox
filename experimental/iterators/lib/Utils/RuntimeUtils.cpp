//===-- RuntimeUtils.cpp - Utils for MLIR / C++ interop ---------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "iterators/Utils/RuntimeUtils.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <tuple>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Operators/ReduceOperator.h"
#include "iterators/Utils/Tuple.h"

using namespace mlir::iterators::operators;
using namespace mlir::iterators::utils;

//===---------------------------------------------------------------------===//
// Exposed C API.
//===---------------------------------------------------------------------===//

extern "C" int8_t *iteratorsMakeSampleInputOperator() {
  std::vector<int32_t> sampleInput = {1, 2, 3};
  using ScanOperatorType = decltype(makeColumnScanOperator(sampleInput));
  return reinterpret_cast<int8_t *>(new ScanOperatorType(sampleInput));
}

extern "C" void iteratorsDestroySampleInputOperator(int8_t *const op) {
  using ScanOperatorType = ColumnScanOperator<int32_t>;
  auto *const typedOp = reinterpret_cast<ScanOperatorType *>(op);
  delete typedOp;
}

// This needs to be global because redefining it in two different places yields
// two different types
auto const sum = [](std::tuple<int32_t> t1, std::tuple<int32_t> t2) {
  return std::make_tuple(std::get<0>(t1) + std::get<0>(t2));
};

extern "C" int8_t *iteratorsMakeReduceOperator(int8_t *const upstream) {
  using ScanOperatorType = ColumnScanOperator<int32_t>;
  using ReduceOperatorType = ReduceOperator<ScanOperatorType, decltype(sum)>;
  return reinterpret_cast<int8_t *>(new ReduceOperatorType(
      reinterpret_cast<ScanOperatorType *>(upstream), sum));
}

extern "C" void iteratorsDestroyReduceOperator(int8_t *const op) {
  using ScanOperatorType = ColumnScanOperator<int32_t>;
  using ReduceOperatorType = ReduceOperator<ScanOperatorType, decltype(sum)>;
  auto *const typedOp = reinterpret_cast<ReduceOperatorType *>(op);
  delete typedOp;
}

extern "C" void iteratorsComsumeAndPrint(int8_t *const upstream) {
  using ScanOperatorType =
      decltype(makeColumnScanOperator(std::declval<std::vector<int32_t>>()));
  using ReduceOperatorType =
      decltype(makeReduceOperator(std::declval<ScanOperatorType *>(), sum));
  auto *const typedUpstream = reinterpret_cast<ReduceOperatorType *>(upstream);
  typedUpstream->open();
  while (auto const tuple = typedUpstream->computeNext()) {
    printTuple(tuple.value());
    std::cout << "\n";
  }
  typedUpstream->close();
}
