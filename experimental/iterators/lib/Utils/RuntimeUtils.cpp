//===- RuntimeUtils.cpp - Utils for MLIR / C++ interop -------------------===//

#include "iterators/Utils/RuntimeUtils.h"

#include <cstdio>
#include <iostream>
#include <list>
#include <memory>
#include <tuple>

#include <llvm/ADT/Any.h>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Operators/ReduceOperator.h"
#include "iterators/Utils/Tuple.h"

using namespace mlir::iterators::operators;
using namespace mlir::iterators::utils;

//===---------------------------------------------------------------------===//
// Exposed C API.
//===---------------------------------------------------------------------===//

/// List of things that will be destructed after the query completed
using Heap = std::list<llvm::Any>;

/// Construct an object of type `T` on the given heap and return a pointer
/// to the contructed object.
template <typename T, typename... Types>
T *emplaceHeap(int8_t *const heap, Types &&...args) {
  auto const typedHeap = reinterpret_cast<Heap *>(heap);
  // Wrap object into `std::shared_ptr` because `llvm::Any` is not copy-
  // constructible
  typedHeap->emplace_back(std::make_shared<T>(std::forward<Types>(args)...));
  auto const ptr = llvm::any_cast<std::shared_ptr<T>>(typedHeap->back());
  return ptr.get();
}

extern "C" RUNTIME_UTILS_EXPORT int8_t *iteratorsMakeHeap() {
  auto const heap = new Heap();
  return reinterpret_cast<int8_t *>(heap);
}

extern "C" RUNTIME_UTILS_EXPORT void iteratorsDestroyHeap(int8_t *const heap) {
  auto const typedHeap = reinterpret_cast<Heap *>(heap);
  delete typedHeap;
}

extern "C" int8_t *iteratorsMakeSampleInputOperator(int8_t *const heap) {
  std::vector<int32_t> sampleInput = {1, 2, 3};
  using ScanOperatorType = decltype(makeColumnScanOperator(sampleInput));
  auto const ptr = emplaceHeap<ScanOperatorType>(heap, sampleInput);
  return reinterpret_cast<int8_t *>(ptr);
}

// This needs to be global because redefining it in two different places yields
// two different types
auto const sum = [](std::tuple<int32_t> t1, std::tuple<int32_t> t2) {
  return std::make_tuple(std::get<0>(t1) + std::get<0>(t2));
};

extern "C" int8_t *iteratorsMakeReduceOperator(int8_t *const heap,
                                               int8_t *const upstream) {
  using ScanOperatorType =
      decltype(makeColumnScanOperator(std::declval<std::vector<int32_t>>()));
  using ReduceOperatorType =
      decltype(makeReduceOperator(std::declval<ScanOperatorType *>(), sum));
  auto const ptr = emplaceHeap<ReduceOperatorType>(
      heap, reinterpret_cast<ScanOperatorType *>(upstream), sum);
  return reinterpret_cast<int8_t *>(ptr);
}

extern "C" void iteratorsComsumeAndPrint(int8_t *const upstream) {
  using ScanOperatorType =
      decltype(makeColumnScanOperator(std::declval<std::vector<int32_t>>()));
  using ReduceOperatorType =
      decltype(makeReduceOperator(std::declval<ScanOperatorType *>(), sum));
  auto const typedUpstream = reinterpret_cast<ReduceOperatorType *>(upstream);
  typedUpstream->open();
  while (auto const tuple = typedUpstream->computeNext()) {
    printTuple(tuple.value());
    std::cout << "\n";
  }
  typedUpstream->close();
}
