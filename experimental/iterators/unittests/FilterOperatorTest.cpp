//===-- FilterOperatorTest.cpp - Unit tests of Filter operator --*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <gtest/gtest.h>

#include <optional>
#include <vector>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Operators/FilterOperator.h"

using namespace mlir::iterators::operators;

TEST(FilterTest, SingleColumn) {
  std::vector<int32_t> numbers = {1, 2, 3, 4, 5, 6};
  auto scan = makeColumnScanOperator(numbers);
  auto filter = makeFilterOperator(
      &scan, [](auto tuple) { return std::get<0>(tuple) % 2 == 0; });
  filter.open();

  // Consume result
  using Result = std::vector<decltype(filter)::OutputTuple>;
  Result result;
  while (const auto tuple = filter.computeNext())
    result.emplace_back(tuple.value());

  // Verify
  Result referenceResult = {{2}, {4}, {6}};
  EXPECT_EQ(result, referenceResult);

  // Check that we can test for the end again
  EXPECT_FALSE(filter.computeNext());

  filter.close();
}

TEST(FilterTest, TwoColumn) {
  std::vector<int32_t> numbers1 = {1, 2, 3, 4, 5, 6, 7};
  std::vector<int32_t> numbers2 = {7, 6, 5, 4, 3, 2, 1};
  auto scan = makeColumnScanOperator(numbers1, numbers2);
  auto filter = makeFilterOperator(&scan, [](auto tuple) {
    return std::get<0>(tuple) == std::get<1>(tuple);
  });
  filter.open();

  // Consume result
  using Result = std::vector<decltype(filter)::OutputTuple>;
  Result result;
  while (const auto tuple = filter.computeNext())
    result.emplace_back(tuple.value());

  // Verify
  Result referenceResult = {{4, 4}};
  EXPECT_EQ(result, referenceResult);

  filter.close();
}
