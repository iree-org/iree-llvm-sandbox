#include <gtest/gtest.h>

#include <tuple>
#include <vector>

#include "iterators/Operators/ColumnScanOperator.h"
#include "iterators/Operators/MapOperator.h"

TEST(MapTest, SingleColumnMap) {
  std::vector<int32_t> numbers = {1, 2};
  auto scan = makeColumnScanOperator(numbers);
  auto map = makeMapOperator(&scan, [](auto tuple) {
    return std::make_tuple(std::get<0>(tuple) + 1);
  });
  map.open();

  decltype(map)::ReturnType tuple;

  // Consume the two values
  tuple = map.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 2);

  tuple = map.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 3);

  // Check that we have reached the end
  tuple = map.computeNext();
  EXPECT_FALSE(tuple);

  // Check that we can test for the end again
  tuple = map.computeNext();
  EXPECT_FALSE(tuple);

  map.close();
}

TEST(MapTest, TypeChangingMap) {
  std::vector<int32_t> numbers = {1, 2};
  auto scan = makeColumnScanOperator(numbers);
  auto map = makeMapOperator(&scan, [](auto tuple) {
    return std::make_tuple(std::get<0>(tuple), std::get<0>(tuple) + 10);
  });
  map.open();

  decltype(map)::ReturnType tuple;

  // Consume the two values
  tuple = map.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 1);
  EXPECT_EQ(std::get<1>(tuple.value()), 11);

  tuple = map.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 2);
  EXPECT_EQ(std::get<1>(tuple.value()), 12);

  // Check that we have reached the end
  tuple = map.computeNext();
  EXPECT_FALSE(tuple);

  // Check that we can test for the end again
  tuple = map.computeNext();
  EXPECT_FALSE(tuple);

  map.close();
}
