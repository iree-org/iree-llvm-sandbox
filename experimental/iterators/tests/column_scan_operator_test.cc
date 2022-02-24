#include <gtest/gtest.h>

#include <vector>

#include "operators/column_scan.h"

TEST(ColumnScanTest, SingleColumn) {
  std::vector<int32_t> Numbers = {1, 2, 3, 4};
  auto Scan = MakeColumnScanOperator(Numbers);
  Scan.open();

  decltype(Scan)::ReturnType Tuple;

  // Consume the four values
  Tuple = Scan.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 1);

  Tuple = Scan.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 2);

  Tuple = Scan.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 3);

  Tuple = Scan.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 4);

  // Check that we have reached the end
  Tuple = Scan.computeNext();
  EXPECT_FALSE(Tuple);

  // Check that we can test for the end again
  Tuple = Scan.computeNext();
  EXPECT_FALSE(Tuple);

  Scan.close();
}

TEST(ColumnScanTest, MultipleColumn) {
  std::vector<int32_t> column1 = {1, 2};
  std::vector<int32_t> column2 = {3, 4};
  auto Scan = MakeColumnScanOperator(column1, column2);
  Scan.open();

  decltype(Scan)::ReturnType Tuple;

  // Consume the two values
  Tuple = Scan.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 1);
  EXPECT_EQ(std::get<1>(Tuple.value()), 3);

  Tuple = Scan.computeNext();
  EXPECT_TRUE(Tuple);
  EXPECT_EQ(std::get<0>(Tuple.value()), 2);
  EXPECT_EQ(std::get<1>(Tuple.value()), 4);

  // Check that we have reached the end
  Tuple = Scan.computeNext();
  EXPECT_FALSE(Tuple);

  Scan.close();
}
