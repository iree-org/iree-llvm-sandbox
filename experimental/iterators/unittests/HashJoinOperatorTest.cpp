//===-- HashJoinOperatorTest.cpp - Unit tests of HashJoin -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <gtest/gtest.h>

#include <algorithm>
#include <optional>
#include <vector>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Operators/HashJoinOperator.h"

using namespace mlir::iterators::operators;

TEST(HashJoinTest, SingleColumnKey) {
  std::vector<int32_t> leftKeys = {1, 2, 1, 2, 5};
  std::vector<int32_t> leftValues = {1, 1, 2, 2, 5};
  std::vector<int32_t> rightKeys = {6, 1, 2, 1, 2};
  std::vector<int32_t> rightValues = {6, 3, 3, 4, 4};

  auto leftScan = makeColumnScanOperator(leftKeys, leftValues);
  auto rightScan = makeColumnScanOperator(rightKeys, rightValues);
  auto hashJoin = makeHashJoinOperator<1>(&leftScan, &rightScan);

  using ResultTuple = decltype(hashJoin)::OutputTuple;

  // Consume result of hashJoin
  hashJoin.open();
  std::vector<ResultTuple> result;
  while (const auto tuple = hashJoin.computeNext())
    result.emplace_back(tuple.value());

  // Compare with correct result
  std::vector<ResultTuple> referenceResult = {
      {1, 1, 3}, {1, 1, 4}, {1, 2, 3}, {1, 2, 4}, //
      {2, 1, 3}, {2, 1, 4}, {2, 2, 3}, {2, 2, 4}};
  std::sort(result.begin(), result.end());
  std::sort(referenceResult.begin(), referenceResult.end());
  EXPECT_EQ(result, referenceResult);

  // Check that we can test for the end again
  EXPECT_FALSE(hashJoin.computeNext());

  hashJoin.close();
}

TEST(HashJoinTest, TwoColumnKey) {
  std::vector<int32_t> leftKeys1 = {1, 1};
  std::vector<int32_t> leftKeys2 = {2, 2};
  std::vector<int32_t> leftValues = {3, 4};
  std::vector<int32_t> rightKeys1 = {1};
  std::vector<int32_t> rightKeys2 = {2};
  std::vector<int32_t> rightValues = {5};

  auto leftScan = makeColumnScanOperator(leftKeys1, leftKeys2, leftValues);
  auto rightScan = makeColumnScanOperator(rightKeys1, rightKeys2, rightValues);
  auto hashJoin = makeHashJoinOperator<2>(&leftScan, &rightScan);

  using ResultTuple = decltype(hashJoin)::OutputTuple;

  // Consume result of hashJoin
  hashJoin.open();
  std::vector<ResultTuple> result;
  while (const auto tuple = hashJoin.computeNext())
    result.emplace_back(tuple.value());
  hashJoin.close();

  // Compare with correct result
  std::vector<ResultTuple> referenceResult = {{1, 2, 3, 5}, {1, 2, 4, 5}};
  std::sort(result.begin(), result.end());
  std::sort(referenceResult.begin(), referenceResult.end());
  EXPECT_EQ(result, referenceResult);
}

TEST(HashJoinTest, NoValueLeft) {
  std::vector<int32_t> leftKeys = {1, 1};
  std::vector<int32_t> rightKeys = {1};
  std::vector<int32_t> rightValues = {2};

  auto leftScan = makeColumnScanOperator(leftKeys);
  auto rightScan = makeColumnScanOperator(rightKeys, rightValues);
  auto hashJoin = makeHashJoinOperator<1>(&leftScan, &rightScan);

  using ResultTuple = decltype(hashJoin)::OutputTuple;

  // Consume result of hashJoin
  hashJoin.open();
  std::vector<ResultTuple> result;
  while (const auto tuple = hashJoin.computeNext())
    result.emplace_back(tuple.value());
  hashJoin.close();

  // Compare with correct result
  std::vector<ResultTuple> referenceResult = {{1, 2}, {1, 2}};
  std::sort(result.begin(), result.end());
  std::sort(referenceResult.begin(), referenceResult.end());
  EXPECT_EQ(result, referenceResult);
}

TEST(HashJoinTest, NoValueRight) {
  std::vector<int32_t> leftKeys = {1, 1};
  std::vector<int32_t> leftValues = {2, 2};
  std::vector<int32_t> rightKeys = {1};

  auto leftScan = makeColumnScanOperator(leftKeys, leftValues);
  auto rightScan = makeColumnScanOperator(rightKeys);
  auto hashJoin = makeHashJoinOperator<1>(&leftScan, &rightScan);

  using ResultTuple = decltype(hashJoin)::OutputTuple;

  // Consume result of hashJoin
  hashJoin.open();
  std::vector<ResultTuple> result;
  while (const auto tuple = hashJoin.computeNext())
    result.emplace_back(tuple.value());
  hashJoin.close();

  // Compare with correct result
  std::vector<ResultTuple> referenceResult = {{1, 2}, {1, 2}};
  std::sort(result.begin(), result.end());
  std::sort(referenceResult.begin(), referenceResult.end());
  EXPECT_EQ(result, referenceResult);
}
