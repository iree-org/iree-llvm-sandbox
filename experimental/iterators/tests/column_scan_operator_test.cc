#include <gtest/gtest.h>

#include <vector>

#include "operators/column_scan.h"

TEST(ColumnScanTest, SingleColumn) {
  std::vector<int32_t> numbers = {1, 2, 3, 4};
  auto scan = makeColumnScanOperator(numbers);
  scan.open();

  decltype(scan)::ReturnType tuple;

  // Consume the four values
  tuple = scan.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 1);

  tuple = scan.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 2);

  tuple = scan.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 3);

  tuple = scan.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 4);

  // Check that we have reached the end
  tuple = scan.computeNext();
  EXPECT_FALSE(tuple);

  // Check that we can test for the end again
  tuple = scan.computeNext();
  EXPECT_FALSE(tuple);

  scan.close();
}

TEST(ColumnScanTest, MultipleColumn) {
  std::vector<int32_t> column1 = {1, 2};
  std::vector<int32_t> column2 = {3, 4};
  auto scan = makeColumnScanOperator(column1, column2);
  scan.open();

  decltype(scan)::ReturnType tuple;

  // Consume the two values
  tuple = scan.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 1);
  EXPECT_EQ(std::get<1>(tuple.value()), 3);

  tuple = scan.computeNext();
  EXPECT_TRUE(tuple);
  EXPECT_EQ(std::get<0>(tuple.value()), 2);
  EXPECT_EQ(std::get<1>(tuple.value()), 4);

  // Check that we have reached the end
  tuple = scan.computeNext();
  EXPECT_FALSE(tuple);

  scan.close();
}
