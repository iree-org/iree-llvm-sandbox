#include <gtest/gtest.h>

#include <map>
#include <tuple>
#include <vector>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Operators/ReduceByKeyOperator.h"

TEST(ReduceByKeyTest, SingleColumnKey) {
  std::vector<int32_t> keys = {1, 2, 1, 2};
  std::vector<int32_t> values = {1, 2, 3, 4};
  auto scan = makeColumnScanOperator(keys, values);
  auto reduceByKey = makeReduceByKeyOperator<1>(&scan, [](auto t1, auto t2) {
    return std::make_tuple(std::get<0>(t1) + std::get<0>(t2));
  });

  // Consume result of reduceByKey
  reduceByKey.open();
  std::map<int32_t, int32_t> result;
  while (const auto tuple = reduceByKey.computeNext()) {
    EXPECT_EQ(result.count(std::get<0>(tuple.value())), 0);
    result.emplace(std::get<0>(tuple.value()), std::get<1>(tuple.value()));
  }

  // Compare with correct result
  std::map<int32_t, int32_t> referenceResult = {{1, 4}, {2, 6}};
  EXPECT_EQ(result, referenceResult);

  // Check that we can test for the end again
  EXPECT_FALSE(reduceByKey.computeNext());

  reduceByKey.close();
}

TEST(ReduceByKeyTest, TwoColumnKey) {
  std::vector<int32_t> keys1 = {1, 1, 1, 1};
  std::vector<int32_t> keys2 = {1, 2, 1, 2};
  std::vector<int32_t> values = {1, 2, 3, 4};
  auto scan = makeColumnScanOperator(keys1, keys2, values);
  auto reduceByKey = makeReduceByKeyOperator<2>(&scan, [](auto t1, auto t2) {
    return std::make_tuple(std::get<0>(t1) + std::get<0>(t2));
  });

  // Consume result of reduceByKey
  using ResultType =
      std::map<std::tuple<int32_t, int32_t>, std::tuple<int32_t>>;
  ResultType result;
  reduceByKey.open();
  while (const auto tuple = reduceByKey.computeNext()) {
    auto const key = takeFront<2>(tuple.value());
    auto const value = dropFront<2>(tuple.value());
    EXPECT_EQ(result.count(key), 0);
    result.emplace(key, value);
  }

  // Compare with correct result
  ResultType referenceResult = {{{1, 1}, {4}}, {{1, 2}, {6}}};
  EXPECT_EQ(result, referenceResult);

  reduceByKey.close();
}
